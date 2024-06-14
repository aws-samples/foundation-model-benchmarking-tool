import os
import sys
import time
import json
import logging
import requests
import subprocess
from typing import Dict
from pathlib import Path
# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# globals
# HF_TOKEN_FNAME: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hf_token.txt")
def set_up(model_name, model_id):    
    directory = os.path.join(os.path.expanduser("~/foundation-model-benchmarking-tool"), model_name)
    os.makedirs(directory, exist_ok=True)
    # Write the content to the serving.properties file
    content =  """\
engine=MPI
option.tensor_parallel_degree=1
option.max_rolling_batch_size=256
option.model_id={model_id}
option.rolling_batch=lmi-dist"""
    file_path = os.path.join(directory, "serving.properties")
    with open(file_path, "w") as file:
        file.write(content)
    print(f"The 'serving.properties' file has been created in {directory}")
    return directory

def create_deplyment_script(image_uri, region, model_name, HF_TOKEN, directory):
# Create the deploy_model.sh script
    deploy_script_content = f"""#!/bin/bash
echo "Going to download model now"
echo "Content in docker command: {region}, {image_uri}, {model_name},{HF_TOKEN}"
aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {image_uri}
docker pull {image_uri}
docker run -d --runtime=nvidia --gpus all --shm-size 12g \\
 -v /home/ubuntu/{model_name}:/opt/ml/model:ro \\
 -v /home/ubuntu/model_server_logs:/opt/djl/logs \\
 -e HF_TOKEN={HF_TOKEN} \\
 -p 8080:8080 \\
 {image_uri}
echo "Done pulling model"
"""
    file_path = os.path.join(directory, "deploy_model.sh")
    with open(file_path, "w") as file:
        file.write(deploy_script_content)
    print(f"The 'deploy_model.sh' file has been created in {directory}")
    return directory

def pull_container(directory):
    """
    Runs the deploy_model.sh bash script with the provided arguments.
    """
    print(f"directory is: {directory}")
    script_path = f"{directory}/deploy_model.sh"  # Replace with the actual path to the script
    print(f"script_path is: {script_path}")
    try:
        subprocess.run(["bash", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running deploy_model.sh script: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def check_model_deployment(endpoint):
    """
    Check the model deployment status and wait for the model to be ready.
    """
    start_time = time.time()
    timeout = 1200  # 20 minutes
    print(f"Checking deployment status at {endpoint}")
    data = {"inputs": ["tell me a story of the little red riding hood"]}
    headers = {"content-type": "application/json"}
    while time.time() - start_time < timeout:
        try:
            response = requests.post(endpoint, headers=headers, json=data)
            logger.info(f"response is: {response.text}")
            if response.status_code == 200:
                print("Model deployment successful!")
                return 0
            else:
                print(f"Model deployment is not ready yet. Return code: {response.status_code}")
                time.sleep(60)  # Wait for 1 minute before checking again
        except Exception as e:
            logger.error(f"Error occurred while deploying the endpoint: {e}")
            time.sleep(60)  # Wait for 1 minute before checking again
    if not deployment_successful:
        print("Timed out waiting for model deployment.")

def deploy(experiment_config: Dict, role_arn: str) -> Dict:
    print("Deploying...")
    image_uri: str=experiment_config['image_uri']
    model_name: str=experiment_config['name']
    logger.info(f"model_name: {model_name}")
    ep_name: str=experiment_config['ep_name']
    model_id: str=experiment_config['model_id']
    region: str=experiment_config['region']
    HF_TOKEN='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'
    try:
        print(ep_name)
        directory = set_up(model_name, model_id)
        directory = create_deplyment_script(image_uri, region, model_name, HF_TOKEN, directory)
        print(f"directory is: {directory}")
        pull_container(directory)
        time.sleep(60)  # Wait for 1 minute before checking again
        check_model_deployment(ep_name)
        return dict(endpoint_name= ep_name, experiment_name=experiment_config['name'])
    except Exception as e:
        print(f"Error during deployment: {e}")
        return {"error": str(e)}
