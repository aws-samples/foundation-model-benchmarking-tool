"""
Deploys a model on AWS silicon

1. Configuration is read from the configured serving.properties file.
2. A hf_token.txt file is required to download the model from Hugging Face.
"""

# Import necessary libraries
import os
import logging
import subprocess
from typing import Dict
from pathlib import Path

# Set up a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global constant for the Hugging Face token file
HF_TOKEN_FNAME: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hf_token.txt")

def deploy(experiment_config: Dict, role_arn: str) -> Dict:
    """
    Function to deploy the model and create the endpoint
    """
    logger.info("Inside neuron deploy function")
    
    try:
        model_id = experiment_config['model_id']
        region = experiment_config['region']
        ml_instance_type = experiment_config['instance_type']
        batch_size = experiment_config['ec2']['batch_size']
        num_neuron_cores = experiment_config['ec2']['num_neuron_cores']
        neuron_version = experiment_config['ec2']['neuron_version']
        model_loading_timeout = experiment_config['ec2']['model_loading_timeout']
        serving_properties = experiment_config['serving.properties']
        s3_bucket = experiment_config['bucket']
        role = experiment_config['sagemaker_execution_role']
        script_path = "./src/fmbench/scripts/compile-llm-for-aws-silicon"
        logger.info("Model ID: %s", model_id)
        logger.info("Region: %s", region)
        logger.info("Instance Type: %s", ml_instance_type)
        logger.info("Batch Size: %s", batch_size)
        logger.info("Number of Neuron Cores: %s", num_neuron_cores)
        logger.info("Neuron Version: %s", neuron_version)
        logger.info("S3 Bucket: %s", s3_bucket)
        logger.info("Role ARN: %s", role)
        logger.info("Script Path: %s", script_path)
        logger.info("IN 1")
        hf_token_file_path = Path(HF_TOKEN_FNAME)
        if hf_token_file_path.is_file() is True:
            logger.info(f"hf_token file path: {hf_token_file_path} is a file")
            HF_TOKEN = Path(HF_TOKEN_FNAME).read_text().strip()
        else:
            logger.info(f"hf_token file path: {hf_token_file_path} is not a file")
        logger.info("HF Token is: %s", HF_TOKEN)

    except KeyError as e:
        logger.info("in 2")
        logger.error("Missing key in experiment_config: %s", e)
        raise
    except FileNotFoundError as e:
        logger.info("in 3")
        logger.error("File not found: %s", e)
        raise
    except Exception as e:
        logger.info(" in 4\n")
        logger.error("Error reading configuration: %s", e)
        raise

    command = [
        "./src/fmbench/scripts/compile-llm-for-aws-silicon/scripts/download_compile_deploy.sh",
        HF_TOKEN,
        model_id,
        neuron_version,
        "model_store",
        s3_bucket,
        "lmi",
        region,
        role,
        batch_size,
        num_neuron_cores,
        ml_instance_type,
        model_loading_timeout,
        serving_properties,
        script_path
    ]
    
    logger.info("Constructed command: %s", command)

    deployment_result = {
        'endpoint_name': None,
        'experiment_name': None,
        'instance_type': None,
        'instance_count': None,
        'deployed': False
    }

    try:
        with open('scripts.log', 'a') as log_file:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logger.info("5\n")
            log_file.write("Standard Output:\n")
            log_file.write(result.stdout)
            log_file.write("\nStandard Error:\n")
            log_file.write(result.stderr)
            
            logger.info("Script executed successfully: %s", result.stdout)
            logger.info("going to read endpoint from endpoint.txt")

            logger.info("going to read endpoint from endpoint.txt")
            
            try:
                # Read the endpoint name from the endpoint.txt file
                logger.info("6\n")
                endpoint_file_path = os.path.join(script_path, "endpoint.txt")
                ep_name = Path(endpoint_file_path).read_text().strip()
                logger.info("Endpoint is: %s", ep_name)
                if ep_name:
                    logger.info("Model endpoint running with name: %s", ep_name)
                    deployment_result.update({
                        'endpoint_name': ep_name,
                        'experiment_name': experiment_config['name'],
                        'instance_type': experiment_config['instance_type'],
                        'instance_count': experiment_config['instance_count'],
                        'deployed': True
                    })
                    logger.info("Deployment result: %s", deployment_result)
                else:
                    logger.error("Model endpoint not running!")
            except FileNotFoundError:
                logger.error("File endpoint.txt not found!")
            except Exception as e:
                logger.error("Error reading endpoint.txt: %s", str(e))
                raise

    except subprocess.CalledProcessError as e:
        with open('scripts.log', 'a') as log_file:
            log_file.write("Standard Error:\n")
            log_file.write(e.stderr)
        
        logger.error("Error executing script: %s", e.stderr)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", str(e))
        raise
    logger.info("Deployment result: %s", deployment_result)
    return deployment_result
