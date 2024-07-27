"""
Deploys a model on AWS silicon

1. Configuration is read from the configured serving.properties file.
2. A hf_token.txt file is required to download the model from Hugging Face.
"""
# Import necessary libraries
import os
import sys
import logging
import subprocess
from typing import Dict
from pathlib import Path

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# globals
HF_TOKEN_FNAME: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hf_token.txt")

def deploy(experiment_config: Dict, role_arn: str) -> Dict:
    """
    Function to deploy the model and create the endpoint
    """
    logger.info("Inside neuron deploy function")
    
    try:
        model_id: str = experiment_config['model_id']
        region: str = experiment_config['region']
        ml_instance_type: str = experiment_config['instance_type']
        batch_size: str = experiment_config['ec2']['batch_size']
        num_neuron_cores: str = experiment_config['ec2']['num_neuron_cores']
        neuron_version: str = experiment_config['ec2']['neuron_version']
        model_loading_timeout: str = experiment_config['ec2']['neuron_version']
        serving_properties: str = experiment_config['serving.properties']
        s3_bucket: str = experiment_config['bucket']
        role: str = experiment_config['sagemaker_execution_role']
        logger.info(HF_TOKEN_FNAME)
        HF_TOKEN: str = Path(HF_TOKEN_FNAME).read_text().strip()
        model_store: str = "model_store"
        s3_prefix: str = "lmi"
        script_path: str = "./src/fmbench/scripts/compile-llm-for-aws-silicon"

        logger.info("Model ID: %s", model_id)
        logger.info("Region: %s", region)
        logger.info("Instance Type: %s", ml_instance_type)
        logger.info("Batch Size: %s", batch_size)
        logger.info("Number of Neuron Cores: %s", num_neuron_cores)
        logger.info("Neuron Version: %s", neuron_version)
        logger.info("S3 Bucket: %s", s3_bucket)
        logger.info("Role ARN: %s", role)
        logger.info("HF Token length: %d", len(HF_TOKEN))
        logger.info("script path: %s", script_path)
        
    except KeyError as e:
        logger.error("Missing key in experiment_config: %s", e)
        raise
    
    except Exception as e:
        logger.error("Error reading configuration: %s", e)
        raise

    # Construct the command
    command = [
        "./src/fmbench/scripts/compile-llm-for-aws-silicon/scripts/download_compile_deploy.sh",
        HF_TOKEN,
        model_id,
        neuron_version,
        model_store,
        s3_bucket,
        s3_prefix,
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

    # Run the command using subprocess.run()
    try:
        with open('scripts.log', 'a') as log_file:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Log the standard output and error to the file
            log_file.write("Standard Output:\n")
            log_file.write(result.stdout)
            log_file.write("\nStandard Error:\n")
            log_file.write(result.stderr)
            
            # Log the output using logger
            logger.info("Script executed successfully: %s", result.stdout)
            return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        # Log the error output
        with open('scripts.log', 'a') as log_file:
            log_file.write("Standard Error:\n")
            log_file.write(e.stderr)
        
        logger.error("Error executing script: %s", e.stderr)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", str(e))
        raise
