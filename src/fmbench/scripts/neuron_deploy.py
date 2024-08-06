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


# Set up a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up a separate logger for script output
script_logger = logging.getLogger('script_logger')
script_logger.setLevel(logging.INFO)

# Create a file handler for logging script output in real time
file_handler = logging.FileHandler('scripts.log', mode='a')
file_handler.setLevel(logging.INFO)

# Create a logging format and add the handler to the script logger
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
script_logger.addHandler(file_handler)

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
        logger.info("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing requirements: {e}")
        raise

    try:
        model_id = experiment_config['model_id']
        region = experiment_config['region']
        ml_instance_type = experiment_config['instance_type']
        batch_size = experiment_config['ec2']['batch_size']
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
        if hf_token_file_path.is_file():
            logger.info(f"hf_token file path: {hf_token_file_path} is a file")
            HF_TOKEN = hf_token_file_path.read_text().strip()
        else:
            logger.error(f"hf_token file path: {hf_token_file_path} is not a file")
            raise FileNotFoundError(f"hf_token.txt file not found at {hf_token_file_path}")
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
        instance_count
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
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Log output and errors in real time
        for stdout_line in process.stdout:
            script_logger.info(stdout_line.strip())
        for stderr_line in process.stderr:
            script_logger.error(stderr_line.strip())
        
        process.stdout.close()
        process.stderr.close()

        return_code = process.wait()
        if return_code != 0:
            logger.error("Script failed with return code: %d", return_code)
            raise subprocess.CalledProcessError(return_code, command)

        logger.info("Script executed successfully, reading endpoint from endpoint.txt")

        # Read the endpoint name from the endpoint.txt file
        try:
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
            raise
        except Exception as e:
            logger.error("Error reading endpoint.txt: %s", str(e))
            raise

    except subprocess.CalledProcessError as e:
        logger.error("Error executing script: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", str(e))
        raise

    logger.info("Deployment result: %s", deployment_result)
    return deployment_result
