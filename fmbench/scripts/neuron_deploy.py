"""
Deploys a model on AWS silicon

1. Configuration is read from the configured serving.properties file.
2. A hf_token.txt file is required to download the model from Hugging Face.
"""

# Import necessary libraries
import os
import sys
import stat
import logging
import subprocess
from typing import Dict
from pathlib import Path
from fmbench.scripts import constants

# Set up a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the platform where this script deploys the model
PLATFORM: str = constants.PLATFORM_SAGEMAKER

# Global constant for the Hugging Face token file
SCRIPT_DIRECTORY: str = os.path.dirname(os.path.realpath(__file__))
HF_TOKEN_FNAME: str = os.path.join(SCRIPT_DIRECTORY, "hf_token.txt")
neuron_script_dir: str = os.path.join(SCRIPT_DIRECTORY, "compile-llm-for-aws-silicon")
shell_script_path: str = os.path.join(neuron_script_dir, "scripts/download_compile_deploy.sh")


def make_executable(file_path: str) -> None:
    """Make the script executable if it is not already."""
    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | stat.S_IEXEC)
    logger.info("Made script executable: %s", file_path)

def deploy(experiment_config: Dict, role_arn: str) -> Dict:
    """
    Function to deploy the model and create the endpoint
    """
    logger.info("Inside neuron deploy function")

    # Ensure the script is executable
    make_executable(shell_script_path)
    
    requirements_file_path = os.path.join(neuron_script_dir, "requirements.txt")
    command = [sys.executable, '-m', 'pip', 'install', '-r', requirements_file_path]

    try:
        subprocess.check_call(command)
        print("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")

    try:
        model_id = experiment_config['model_id']
        region = experiment_config['region']
        ml_instance_type = experiment_config['instance_type']
        batch_size = experiment_config['ec2']['batch_size']
        image_uri = experiment_config['image_uri']
        num_neuron_cores = experiment_config['ec2']['num_neuron_cores']
        neuron_version = experiment_config['ec2']['neuron_version']
        model_loading_timeout = experiment_config['ec2']['model_loading_timeout']
        prefix = experiment_config['prefix']
        serving_properties = experiment_config['serving.properties']
        s3_bucket = experiment_config['bucket']
        role = experiment_config['sagemaker_execution_role']
        instance_count = experiment_config['ec2']['instance_count']

        logger.info("Model ID: %s", model_id)
        logger.info("Region: %s", region)
        logger.info("Instance Type: %s", ml_instance_type)
        logger.info("Batch Size: %s", batch_size)
        logger.info("Number of Neuron Cores: %s", num_neuron_cores)
        logger.info("Neuron Version: %s", neuron_version)
        logger.info("S3 Bucket: %s", s3_bucket)
        logger.info("Role ARN: %s", role)
        logger.info("Script Path: %s", neuron_script_dir)
        logger.info("Model Loading Timeout: %s", model_loading_timeout)
        logger.info("Initial Instance Count: %s", instance_count)
        hf_token_file_path = Path(HF_TOKEN_FNAME)
        if hf_token_file_path.is_file() is True:
            logger.info(f"hf_token file path: {hf_token_file_path} is a file")
            HF_TOKEN = Path(HF_TOKEN_FNAME).read_text().strip()
        else:
            logger.info(f"hf_token file path: {hf_token_file_path} is not a file")
        logger.info("HF Token is: %s", HF_TOKEN)

    except KeyError as e:
        logger.error("Missing key in experiment_config: %s", e)
        raise
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error reading configuration: %s", e)
        raise

    logger.info(f"Image uri that is being used: {image_uri}")
    command = [
        shell_script_path,
        HF_TOKEN,
        model_id,
        neuron_version,
        "model_store",
        s3_bucket,
        prefix,
        region,
        role,
        batch_size,
        num_neuron_cores,
        ml_instance_type,
        model_loading_timeout,
        serving_properties,
        neuron_script_dir,
        instance_count, 
        image_uri
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
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    
            for line in process.stdout:
                log_file.write(f"Standard Output: {line}")
                log_file.flush()
                logger.info("Script output: %s", line.strip())

            for line in process.stderr:
                log_file.write(f"Standard Error: {line}")
                log_file.flush()
                logger.error("Script error: %s", line.strip())
            return_code = process.wait()
            
            if return_code == 0:
                logger.info("Script executed successfully")
            else:
                logger.error("Script failed with return code: %d", return_code)
            
            logger.info("Going to read endpoint from endpoint.txt")
            try:
                # Read the endpoint name from the endpoint.txt file
                endpoint_file_path = os.path.join(neuron_script_dir, "endpoint.txt")
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
