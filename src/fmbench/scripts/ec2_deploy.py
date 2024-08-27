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
import stat
import docker
import logging
import requests
import tempfile
import subprocess
from typing import Dict
from pathlib import Path
from fmbench.scripts import constants
from ec2_metadata import ec2_metadata


# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# globals
HF_TOKEN_FNAME: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hf_token.txt")
# SHM_SIZE: str = "12g"
# MODEL_DEPLOYMENT_TIMEOUT: int = 2400
FMBENCH_MODEL_CONTAINER_NAME: str = "fmbench_model_container"

def _set_up(model_name: str, serving_properties: str, local_model_path: str):
    """
    Create the model serving.properties file locally in a model directory
    """
    # make the model directory with serving.properties
    directory = os.path.join(local_model_path, model_name)
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Creating the local directory: {directory}")
    if serving_properties:
        file_path = os.path.join(directory, "serving.properties")
        Path(file_path).write_text(serving_properties)
        logger.info(f"The serving.properties file has been created in {file_path}")
    else:
        logger.info(f"no serving.properties content specified, not creating the file")

    #return the directory we created
    return directory

def _create_deployment_script(image_uri,
                              container_type,
                              privileged_mode,
                              region,
                              model_name,
                              model_id,
                              HF_TOKEN,
                              directory,
                              gpu_or_neuron_setting,
                              model_loading_timeout,
                              env):
    """
    Write a deployment script for model container
    """
    #stop container if it already exists check if container exists 
    container_name: str = FMBENCH_MODEL_CONTAINER_NAME
    env_str: str = f"-e MODEL_LOADING_TIMEOUT={model_loading_timeout} "
    if privileged_mode:
        privileged_flag = "--privileged"
    else:
        privileged_flag = ""
    if env is not None:
        logger.info(f"env passed is: {env}")
    
        for env_dict in env:
            for k, v in env_dict.items():
                env_str += f"-e {k}={v} "

    if container_type == constants.CONTAINER_TYPE_DJL:
        deploy_script_content = f"""#!/bin/sh
        echo "Going to download model now"
        echo "Content in docker command: {region}, {image_uri}, {model_name}"

        # Login to AWS ECR and pull the Docker image
        aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {image_uri}
        docker pull {image_uri}

        # Attempt to stop and remove the container up to 3 times if container exists
        if [ -n "$(docker ps -aq --filter "name={container_name}")" ]; then
            for i in {{1..3}}; do
                echo "Attempt $i to stop and remove the container: {container_name}"
                
                # Stop the container
                docker ps -q --filter "name={container_name}" | xargs -r docker stop
                
                # Wait for 5 seconds
                sleep 5
                
                # Remove the container
                docker ps -aq --filter "name={container_name}" | xargs -r docker rm
                
                # Wait for 5 seconds
                sleep 5
                
                # Check if the container is removed
                if [ -z "$(docker ps -aq --filter "name={container_name}")" ]; then
                    echo "Container {container_name} successfully stopped and removed."
                    break
                else
                    echo "Container {container_name} still exists, retrying..."
                fi
            done
        else
            echo "Container {container_name} does not exist. No action taken."
        fi

        # Run the new Docker container with specified settings
        docker run -d --name={container_name} {gpu_or_neuron_setting} \
            -v {directory}:/opt/ml/model:ro \
            -v {directory}/model_server_logs:/opt/djl/logs \
            {env_str} \
            -e HF_TOKEN={HF_TOKEN} \
            -p 8080:8080 {image_uri}

        echo "Done pulling model"

        """
    elif container_type == constants.CONTAINER_TYPE_VLLM:
        deploy_script_content = f"""#!/bin/sh
        Attempt to stop and remove the container up to 3 times if container exists
        if [ -n "$(docker ps -aq --filter "name={container_name}")" ]; then
            for i in {{1..3}}; do
                echo "Attempt $i to stop and remove the container: {container_name}"
                
                # Stop the container
                docker ps -q --filter "name={container_name}" | xargs -r docker stop
                
                # Wait for 5 seconds
                sleep 5
                
                # Remove the container
                docker ps -aq --filter "name={container_name}" | xargs -r docker rm
                
                # Wait for 5 seconds
                sleep 5
                
                # Check if the container is removed
                if [ -z "$(docker ps -aq --filter "name={container_name}")" ]; then
                    echo "Container {container_name} successfully stopped and removed."
                    break
                else
                    echo "Container {container_name} still exists, retrying..."
                fi
            done
        else
            echo "Container {container_name} does not exist. No action taken."
        fi

        # Run the new Docker container with specified settings
        docker run -d {privileged_flag} --rm --name={container_name} --env "HF_TOKEN={HF_TOKEN}" --ipc=host -p 8000:8000 {env_str} {image_uri} --model {model_id}

        echo "Done running {image_uri} image"

"""
    script_file_path = os.path.join(directory, "deploy_model.sh")
    Path(script_file_path).write_text(deploy_script_content)
    logger.info(f"deploy_model.sh content: {deploy_script_content}")
    logger.info(f"The deploy_model.sh file has been created in {script_file_path}")
    return script_file_path

def _run_container(script_file_path):
    """
    Runs the deploy_model.sh bash script with the provided arguments.
    """
    logger.info(f"Running container script at {script_file_path}")
    # Create a Docker client
    client = docker.from_env()

    try:
        # Check if the container exists and is running
        logger.info(f"going to check if the {FMBENCH_MODEL_CONTAINER_NAME} container is running")
        container = client.containers.get(FMBENCH_MODEL_CONTAINER_NAME)
        logger.info(f"after checking if the {FMBENCH_MODEL_CONTAINER_NAME} container is running")
        if container.status == "running":
            logger.info(f"Container {FMBENCH_MODEL_CONTAINER_NAME} is already running.")
        else:
            logger.info(f"Container {FMBENCH_MODEL_CONTAINER_NAME} is not running.")
        subprocess.run(["bash", script_file_path], check=True)
        logger.info(f"done running bash script")
        return True
    except docker.errors.NotFound:
        logger.info(f"Container {FMBENCH_MODEL_CONTAINER_NAME} not found. Running the script directly.")
        subprocess.run(["bash", script_file_path], check=True)
        logger.info(f"done running bash script")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running deploy_model.sh script: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


    return False

def _check_model_deployment(endpoint, model_id, container_type, model_loading_timeout):
    """
    Check the model deployment status and wait for the model to be ready.
    """
    start_time = time.time()
    #global variable 
    timeout = model_loading_timeout
    logger.info(f"Checking deployment status at {endpoint}")
    if container_type == constants.CONTAINER_TYPE_DJL:
        data = {"inputs": ["tell me a story of the little red riding hood"]}
    elif container_type == constants.CONTAINER_TYPE_VLLM:
        data = {"model": model_id,  # Specify the model to use
                "prompt": "tell me a story of the little red riding hood",}
    headers = {"content-type": "application/json"}
    while time.time() - start_time < timeout:
        time.sleep(60)
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
    privileged_mode: str = experiment_config['ec2'].get('privileged_mode', "")
    container_type: str = experiment_config['inference_spec'].get('container_type', constants.CONTAINER_TYPE_DJL)
    env = experiment_config.get('env')
    serving_properties: str = experiment_config['serving.properties']
    model_loading_timeout: int = experiment_config['ec2']['model_loading_timeout']
    
    gpu_or_neuron_setting: str = experiment_config['ec2']['gpu_or_neuron_setting']
    HF_TOKEN: str = Path(HF_TOKEN_FNAME).read_text().strip()
    dir_path = home_dir = os.getenv("HOME", str(Path.home()))
    logger.info(f"Home directory: {dir_path}")
    
    model_directory = _set_up(model_name, serving_properties, dir_path)    
    
    logger.info("Creating the deployment script in model directory")
    deployment_script_path = _create_deployment_script(image_uri,
                                                       container_type,
                                                       privileged_mode,
                                                       region,
                                                       model_name,
                                                       model_id,
                                                       HF_TOKEN,
                                                       model_directory,
                                                       gpu_or_neuron_setting,
                                                       model_loading_timeout,
                                                       env)

    logger.info("Running the deployment script")
    ran_container = _run_container(deployment_script_path)

    # print the ec2 instance details for it to get logged
    logger.info(f"EC2 instance type: {ec2_metadata.instance_type}, AMI ID: {ec2_metadata.ami_id}")
    # initialize with None values for error case
    deployment_result: Dict = dict(endpoint_name=None, 
                        experiment_name=None,
                        instance_type=None,
                        instance_count=None, 
                        deployed=False)
    if ran_container:
        logger.info("Container ran successfully")
        ep_status = _check_model_deployment(ep_name, model_id, container_type, model_loading_timeout)
        logger.info(f"Endpoint status: {ep_status}")
        if ep_status == "InService":
            logger.info("Model endpoint running!")
            deployment_result['endpoint_name'] = ep_name
            deployment_result['experiment_name'] = experiment_config['name']
            deployment_result['instance_type'] = experiment_config['instance_type']
            deployment_result['instance_count'] = experiment_config['instance_count']
            deployment_result['deployed'] = True
            return deployment_result
        elif ep_status == "Failed":
            logger.error("Model endpoint not running!")
            return deployment_result
    else:
        logger.error("Container did not run successfully")
        return deployment_result
