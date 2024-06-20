"""
Deploys a model from HuggingFace on Amazon EC2

1. Configuration is read from the configured serving.properties file.
2. A hf_token.txt file is required to download the model from Hugging Face.
"""
# Import necessary libraries
import os
import sys
import time
import json
import logging
import requests
import tempfile
import subprocess
from typing import Dict
from pathlib import Path

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# globals
HF_TOKEN_FNAME: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hf_token.txt")
SHM_SIZE: str = "12g"
MODEL_DEPLOYMENT_TIMEOUT: int = 2400

def _set_up(model_name: str, serving_properties: str, local_model_path: str):
    """
    Create the model serving.properties file locally in a model directory
    """
    # make the model directory with serving.properties
    directory = os.path.join(local_model_path, model_name)
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Creating the local directory: {directory}")

    file_path = os.path.join(directory, "serving.properties")
    Path(file_path).write_text(serving_properties)
    logger.info(f"The serving.properties file has been created in {file_path}")
    return file_path

def _create_deployment_script(image_uri, region, model_name, HF_TOKEN, directory):
# Create the deploy_model.sh script
    deploy_script_content = f"""#!/bin/bash
echo "Going to download model now"
echo "Content in docker command: {region}, {image_uri}, {model_name},{HF_TOKEN}"
aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {image_uri}
docker pull {image_uri}
docker run -d --runtime=nvidia --gpus all --shm-size {SHM_SIZE} \\
 -v /home/ubuntu/{model_name}:/opt/ml/model:ro \\
 -v /home/ubuntu/model_server_logs:/opt/djl/logs \\
 -e HF_TOKEN={HF_TOKEN} \\
 -p 8080:8080 \\
 {image_uri}
echo "Done pulling model"
"""
    script_file_path = os.path.join(directory, "deploy_model.sh")
    Path(script_file_path).write_text(deploy_script_content)
    logger.info(f"The 'deploy_model.sh' file has been created in {script_file_path}")
    return script_file_path

def _run_container(script_file_path):
    """
    Runs the deploy_model.sh bash script with the provided arguments.
    """
    logger.info(f"Running container script at {script_file_path}")
    try:
        subprocess.run(["bash", script_file_path], check=True)
        logger.info(f"done running bash script")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running deploy_model.sh script: {e}")
    except Exception as e:
       logger.error(f"An unexpected error occurred: {e}")
    return False

def _check_model_deployment(endpoint):
    """
    Check the model deployment status and wait for the model to be ready.
    """
    start_time = time.time()
    #global variable 
    timeout = MODEL_DEPLOYMENT_TIMEOUT
    logger.info(f"Checking deployment status at {endpoint}")
    data = {"inputs": ["tell me a story of the little red riding hood"]}
    headers = {"content-type": "application/json"}
    while time.time() - start_time < timeout:
        try:
            response = requests.post(endpoint, headers=headers, json=data)
            logger.info(f"response is: {response.text}")
            if response.status_code == 200:
                logger.info("Model deployment successful!")
                return "InService"
            else:
                print(f"Model deployment is not ready yet. Return code: {response.status_code}")
                time.sleep(60)  # Wait for 1 minute before checking again
        except Exception as e:
            logger.error(f"Error occurred while deploying the endpoint: {e}")
            time.sleep(60)  # Wait for 1 minute before checking again
    return "Failed"

def deploy(experiment_config: Dict, role_arn: str) -> Dict:
    """
    Function to deploy the model and create the endpoint
    """
    image_uri: str = experiment_config['image_uri']
    model_name: str = experiment_config['name']
    logger.info(f"Going to deploy model: {model_name}")
    ep_name: str = experiment_config['ep_name']
    model_id: str = experiment_config['model_id']
    region: str = experiment_config['region']
    serving_properties: str = experiment_config['serving.properties']
    HF_TOKEN: str = Path(HF_TOKEN_FNAME).read_text().strip()
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    serving_properties_path = _set_up(model_name, serving_properties, dir_path)
    logger.info(f"Writing serving.properties {serving_properties} to {serving_properties_path}")

    logger.info("Creating the deployment script")
    deployment_script_path = _create_deployment_script(image_uri, region, model_name, HF_TOKEN, dir_path)

    logger.info("Running the deployment script")
    ran_container = _run_container(deployment_script_path)

    if ran_container:
        logger.info("Container ran successfully")
        ep_status = _check_model_deployment(ep_name)
        logger.info(f"Endpoint status: {ep_status}")
        if ep_status == "InService":
            logger.info("Model endpoint running!")
            return dict(endpoint_name=ep_name, experiment_name=experiment_config['name'])
        elif ep_status == "Failed":
            logger.error("Model endpoint not running!")
            return dict(endpoint_name=None, experiment_name=None)
    else:
        logger.error("Container did not run successfully")
        return dict(endpoint_name=None, experiment_name=None)