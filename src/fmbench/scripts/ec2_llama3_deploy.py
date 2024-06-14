import os
import sys
import time
import json
import logging
import requests
import subprocess
from typing import Dict

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# globals
# HF_TOKEN_FNAME: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hf_token.txt")
# PORT=8080
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

def pull_container(image_uri, region, model_name, hf_token, directory):
    """
    Runs the deploy_model.sh bash script with the provided arguments.
    """
    directory = os.path.join(os.path.expanduser("~/foundation-model-benchmarking-tool"), "src/fmbench/scripts")
    script_path = f"{directory}/deploy_model.sh"  # Replace with the actual path to the script
    env = {
        "REGION": region,
        "IMAGE_URI": image_uri,
        "MODEL_NAME": model_name,
        "HF_TOKEN": hf_token,
    }

    try:
        print(env)
        print(script_path)
        subprocess.run(["bash", script_path], env=env, check=True)
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
    print(f"endpoint is {endpoint}")
    while time.time() - start_time < timeout:
        try:
            response = requests.get(endpoint)
            logger.info(f"response is: {response}")
            if response.status_code == 200:
                print("Model deployment successful!")
                return 0
            else:
                print(f"Model deployment not ready yet. Status code: {response.status_code}")
                time.sleep(60)  # Wait for 1 minute before checking again
        except requests.exceptions.RequestException as e:
            print(f"Error checking model deployment: {e}")
            time.sleep(60)  # Wait for 1 minute before checking again

    print("Timed out waiting for model deployment.")

def deploy(experiment_config: Dict, role_arn: str) -> Dict:
    print("Deploying...")
    image_uri: str=experiment_config['image_uri']
    model_name: str=experiment_config['name']
    ep_name: str=experiment_config['ep_name']
    model_id: str=experiment_config['model_id']
    region: str="us-east-1"
    model_name: str=experiment_config['name']
    ep_name: str=experiment_config['ep_name']
    model_id: str=experiment_config['model_id']
    HF_TOKEN='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'

    try:
        print("setting up now")  
        directory = set_up(model_name, model_id)
        logger.info(f"directory={directory}")
        print("pulling container now")
        pull_container(image_uri, region, model_name, HF_TOKEN, directory)
        print("Checking model readiness")    
        check_model_deployment(ep_name)
    except Exception as e:
        print(f"Error during deployment: {e}")
        return {"error": str(e)}
