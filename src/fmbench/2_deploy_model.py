#!/usr/bin/env python
# coding: utf-8

# ## Deploy Jumpstart and Non Jumpstart Models Asynchronously 
# ---------------------
# *This notebook works best with the conda_python3 kernel on a ml.t3.medium machine*.
# 
# **This step of our solution design covers setting up the environment, downloading the requirements needed to run the environment, as well as deploying the model endpoints from the config.yml file asychronously.**
# 1. Prerequisite: Navigate to the file: 0_setup.ipynb and Run the cell to import and download the requirements.txt.
# 2. Now you can run this notebook to deploy the models asychronously in different threads. The key components of this notebook for the purposes of understanding are:
# - Loading the globals.py and config.yml file.
# - Setting a blocker function deploy_model to deploy the given model endpoint followed by:
# - A series of async functions to set tasks to deploy the models from the config yml file asynchronously in different threads. View the notebook from the link above.
# - Once the endpoints are deployed, their model configurations are stored within the endpoints.json file.

#### Import all of the necessary libraries below to run this notebook
import sys
import time
import json
import boto3
import asyncio
import logging
import importlib.util
import fmbench.scripts
from pathlib import Path
from fmbench.utils import *
from fmbench.globals import *
from typing import Dict, List, Optional
from sagemaker import get_execution_role
import importlib.resources as pkg_resources
from botocore.exceptions import ClientError
from botocore.exceptions import NoCredentialsError


# #### Pygmentize globals.py to view and use any of the globally initialized variables 

# #### Set up a logger to log all messages while the code runs
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# ### Load the config.yml file
# ------
# 
# The config.yml file contains information that is used across this benchmarking environment, such as information about the aws account, prompts, payloads to be used for invocations, and model configurations like the version of the model, the endpoint name, model_id that needs to be deployed. Configurations also support the gives instance type to be used, for example: "ml.g5.24xlarge", the image uri, whether or not to deploy this given model, followed by an inference script "jumpstart.py" which supports the inference script for jumpstart models to deploy the model in this deploy notebook. 
# 
# View the contents of the config yml file below and how it is loaded and used throughout this notebook with deploying the model endpoints asynchronously.
## Load the config.yml file referring to the globals.py file
config = load_config(CONFIG_FILE)

## configure the aws region and execution role
aws_region = config['aws']['region']


try:
    sagemaker_execution_role = get_execution_role()
    config['aws']['sagemaker_execution_role'] = sagemaker_execution_role
    logger.info(f"determined SageMaker exeuction role from get_execution_role")
except Exception as e:
    logger.error(f"could not determine SageMaker execution role, error={e}")
    logger.info(f"going to look for execution role in config file..")
    sagemaker_execution_role = config['aws'].get('sagemaker_execution_role')
    if sagemaker_execution_role is not None:
        logger.info(f"found SageMaker execution role in config file..")

logger.info(f"aws_region={aws_region}, sagemaker_execution_role={sagemaker_execution_role}")
logger.info(f"config={json.dumps(config, indent=2)}")


# #### Deploy a single model: blocker function used for asynchronous deployment
# This function is designed to deploy a single large language model endpoint. It takes three parameters: experiment_config (a dictionary containing configuration details for the model deployment from the config.yml file), aws_region (the AWS region where the model will be deployed), and role_arn (the AWS role's Amazon Resource Name used for the deployment).
# function to deploy a model
def deploy_model(experiment_config: Dict, aws_region: str, role_arn: str) -> Optional[Dict]:
    
    # Log the deployment details
    logger.info(f"going to deploy {experiment_config}, in {aws_region} with {role_arn}")
    model_deployment_result = None
    
    # Check if deployment is enabled in the config; skip if not
    deploy = experiment_config.get('deploy', False)
    if deploy is False:
        logger.error(f"skipping deployment of {experiment_config['model_id']} because deploy={deploy}")
        return model_deployment_result
    
    # Initialize the S3 client
    s3_client = boto3.client('s3', region_name=aws_region)

    # Assuming fmbench is a valid Python package and scripts is a subdirectory within it
    scripts_dir = Path(pkg_resources.files('fmbench'), 'scripts')
    logger.info(f"Using fmbench.scripts directory: {scripts_dir}")

    # Ensure the scripts directory exists
    scripts_dir.mkdir(parents=True, exist_ok=True)

    read_bucket = config['s3_read_data']['read_bucket']
    logger.info(f"the read bucket is --> {read_bucket} for reading the script files")
    scripts_prefix = config['s3_read_data']['scripts_prefix']
    logger.info(f"the scripts directory is --> {scripts_prefix} for reading the script file names")
    script_files = config['s3_read_data'].get('script_files', [])
    logger.info(f"Extracted script files that the user has provided --> {script_files}")

    # Download script files to the fmbench.scripts directory
    try:
        for script_name in script_files:
            # do os.path.join
            s3_script_path = f"{scripts_prefix}/{script_name}"
            ## take this out of the loop 
            logger.info(f"the script path for where the scripts you have entered in s3 will be installed --> {s3_script_path}")
            local_script_path = scripts_dir / script_name
            logger.info(f"Downloading {s3_script_path} to {local_script_path}")
            s3_client.download_file(read_bucket, s3_script_path, str(local_script_path))
    except ClientError as error:
        logger.error(f"Failed to download script files: {error}")

    # Proceed with deployment as before
    try:
        module_name = Path(experiment_config['deployment_script']).stem
        logger.info(f"The given script provided for inference of this model is --> {module_name}")
        deployment_script_path = scripts_dir / f"{module_name}.py"
        logger.info(f"Deployment script path is --> {deployment_script_path}")

        # Check and proceed with local script
        if not deployment_script_path.exists():
            logger.error(f"Deployment script {deployment_script_path} not found.")
            return None

        logger.info(f"Deploying using local code: {deployment_script_path}")

        spec = importlib.util.spec_from_file_location(module_name, str(deployment_script_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        model_deployment_result = module.deploy(experiment_config, role_arn)
        return model_deployment_result
    

    except Exception as error:  # Broader exception handling for non-ClientError issues
        logger.error(f"An error occurred during deployment: {error}")
        return model_deployment_result


# ### Asynchronous Model Deployment
# ----
# #### async_deploy_model: 
# - This is an asynchronous wrapper around the deploy_model function. It uses asyncio.to_thread to run the synchronous deploy_model function in a separate thread. This allows the function to be awaited in an asynchronous context, enabling concurrent model deployments without any blocking from the main thread
# async_deploy_all_models Function: 
# - This 'async_deploy_all_models' function is designed to deploy multiple models concurrently. It splits the models into batches and deploys each batch concurrently using asyncio.gather.
    

## Asynchronous wrapper function to allow our deploy_model function to allow concurrent requests for deployment
async def async_deploy_model(experiment_config: Dict, role_arn: str, aws_region: str) -> str:
    # Run the deploy_model function in a separate thread to deploy the models asychronously
    return await asyncio.to_thread(deploy_model, experiment_config, role_arn, aws_region)

## Final asychronous function to deploy all of the models concurrently
async def async_deploy_all_models(config: Dict) -> List[Dict]:
    
    ## Extract experiments from the config.yml file (contains information on model configurations)
    experiments: List[Dict] = config['experiments']
    n: int = 4 # max concurrency so as to not get a throttling exception
    
    ## Split experiments into smaller batches for concurrent deployment
    experiments_splitted = [experiments[i * n:(i + 1) * n] for i in range((len(experiments) + n - 1) // n )]
    results = []
    for exp_list in experiments_splitted:
        
        ## send the deployment in batches
        result = await asyncio.gather(*[async_deploy_model(m,
                                                           config['aws']['region'],
                                                           config['aws']['sagemaker_execution_role']) for m in exp_list])
        ## Collect and furthermore extend the results from each batch
        results.extend(result)
    return results

# async version
s = time.perf_counter() ## start the counter to deploy the models asynchronously

## Call all of the models for deployment using the config.yml file model configurations
# endpoint_names = await async_deploy_all_models(config)
endpoint_names = asyncio.run(async_deploy_all_models(config))

## Set a timer for model deployment counter
elapsed_async = time.perf_counter() - s
print(f"endpoint_names -> {endpoint_names}, deployed in {elapsed_async:0.2f} seconds")

## Function to get all of the information on the deployed endpoints and store it in a json
def get_all_info_for_endpoint(ep: Dict) -> Dict:
    
    ## extract the endpoint name
    ep_name = ep['endpoint_name']
    
    ## extract the experiment name from the config.yml file
    experiment_name = ep['experiment_name']
    if ep_name is None:
        return None
    sm_client = boto3.client('sagemaker')
    
    ## get the description on the configuration of the deployed model
    endpoint = sm_client.describe_endpoint(EndpointName=ep_name)
    endpoint_config = sm_client.describe_endpoint_config(EndpointConfigName=endpoint['EndpointConfigName'])
    model_config = sm_client.describe_model(ModelName=endpoint_config['ProductionVariants'][0]['ModelName'])
    
    ## Store the experiment name and all of the other model configuration information in the 'info' dict
    info = dict(experiment_name=experiment_name,
                endpoint=endpoint,
                endpoint_config=endpoint_config,
                model_config=model_config)
    return info

all_info = list(map(get_all_info_for_endpoint, [ep for ep in endpoint_names if ep is not None]))

## stores information in a dictionary for collectively all of the deployed model endpoints
all_info

# Convert data to JSON
json_data = json.dumps(all_info, indent=2, default=str)

# Specify the file name
file_name = "endpoints.json"

# Write to S3
endpoint_s3_path = write_to_s3(json_data, config['aws']['bucket'], MODELS_DIR, "", file_name)

logger.info(f"The s3 endpoints that are deployed are sent to this file --> {endpoint_s3_path}")

