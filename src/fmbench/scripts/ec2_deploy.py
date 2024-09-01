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
from pathlib import Path
from typing import Dict, Union
from fmbench.scripts import constants
from ec2_metadata import ec2_metadata
from fmbench.scripts.prepare_for_multi_model_djl import prepare_for_neuron

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# globals
HF_TOKEN_FNAME: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hf_token.txt")
# SHM_SIZE: str = "12g"
# MODEL_DEPLOYMENT_TIMEOUT: int = 2400
FMBENCH_MODEL_CONTAINER_NAME: str = "fmbench_model_container"

STOP_AND_RM_CONTAINER = f"""
    # Attempt to stop and remove the container up to 3 times if container exists
        if [ -n "$(docker ps -aq --filter "name={FMBENCH_MODEL_CONTAINER_NAME}")" ]; then
            for i in {{1..3}}; do
                echo "Attempt $i to stop and remove the container: {FMBENCH_MODEL_CONTAINER_NAME}"
                
                # Stop the container
                docker ps -q --filter "name={FMBENCH_MODEL_CONTAINER_NAME}" | xargs -r docker stop
                
                # Wait for 5 seconds
                sleep 5
                
                # Remove the container
                docker ps -aq --filter "name={FMBENCH_MODEL_CONTAINER_NAME}" | xargs -r docker rm
                
                # Wait for 5 seconds
                sleep 5
                
                # Check if the container is removed
                if [ -z "$(docker ps -aq --filter "name={FMBENCH_MODEL_CONTAINER_NAME}")" ]; then
                    echo "Container {FMBENCH_MODEL_CONTAINER_NAME} successfully stopped and removed."
                    break
                else
                    echo "Container {FMBENCH_MODEL_CONTAINER_NAME} still exists, retrying..."
                fi
            done
        else
            echo "Container {FMBENCH_MODEL_CONTAINER_NAME} does not exist. No action taken."
        fi
    """

def _set_up(model_name: str, local_model_path: str):
    """
    Create the model serving.properties file locally in a model directory
    """
    # make the model directory with serving.properties
    directory = os.path.join(local_model_path, model_name)
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Creating the local directory: {directory}")
    

    #return the directory we created
    return directory

def _create_script_djl(region, image_uri, model_name, directory, env_str, gpu_or_neuron_setting):
    script = f"""#!/bin/sh
        echo "Going to download container now"
        echo "Content in docker command: {region}, {image_uri}, {model_name}"

        # Login to AWS ECR and pull the Docker image
        aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {image_uri}
        docker pull {image_uri}

        {STOP_AND_RM_CONTAINER}

        # Run the new Docker container with specified settings
        docker run -d --name={CONTAINER_NAME} {gpu_or_neuron_setting} \
            -v {directory}:/opt/ml/model:ro \
            -v {directory}/model_server_logs:/opt/djl/logs \
            {env_str} \
            -e HF_TOKEN={HF_TOKEN} \
            -p 8080:8080 {image_uri}

        echo "started docker run in daemon mode"

    """
    return script

def _create_script_djl_w_docker_compose(region, image_uri, model_name, directory):
    script = f"""#!/bin/sh
        echo "Going to download model container"
        echo "Content in docker command: {region}, {image_uri}, {model_name}"

        # Login to AWS ECR and pull the Docker image
        aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {image_uri}
        docker pull {image_uri}       

        # Run the new Docker container with specified settings
        cd {directory}
        # shutdown existing docker compose
        docker compose down
        {STOP_AND_RM_CONTAINER}
        docker compose up -d
        cd -
        echo "started docker compose in daemon mode"
    """
    return script

def _create_script_vllm(image_uri, model_id, env_str, privileged_str):
    script = f"""#!/bin/sh

        {STOP_AND_RM_CONTAINER}

        # Run the new Docker container with specified settings
        docker run -d {privileged_str} --rm --name={CONTAINER_NAME} --env "HF_TOKEN={HF_TOKEN}" --ipc=host -p 8000:8000 {env_str} {image_uri} --model {model_id}

        echo "started docker run in daemon mode"
    """
    return script

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
                              env,
                              model_copies):
    """
    Write a deployment script for model container
    """
    
    privileged_str: str = "--privileged" if privileged_mode else ""
    env_str: str = ""
    if env is not None:
        for k, v in env.items():
            env_str += f"-e {k}={v} "

    if container_type == constants.CONTAINER_TYPE_DJL:
        if model_copies == "1":
            deploy_script_content = _create_script_djl(region, image_uri, model_name, directory, env_str, gpu_or_neuron_setting)
        else:            
            logger.info(f"going to create docker compose run script as model_copies={model_copies}")
            deploy_script_content = _create_script_djl_w_docker_compose(region, image_uri, model_name, directory)

    elif container_type == constants.CONTAINER_TYPE_VLLM:
        deploy_script_content = _create_script_vllm(image_uri, model_id, env_str, privileged_str)

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
    logger.info(f"waiting for containers to come up...")
    time.sleep(820)
    logger.info(f"after waiting for containers to come up...")
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
    privileged_mode: str = experiment_config['ec2'].get('privileged_mode', False)
    container_type: str = experiment_config['inference_spec'].get('container_type', constants.CONTAINER_TYPE_DJL)
    env = experiment_config.get('env')
    serving_properties: str = experiment_config['serving.properties']
    model_loading_timeout: int = experiment_config['ec2']['model_loading_timeout']
    
    gpu_or_neuron_setting: str = experiment_config['ec2']['gpu_or_neuron_setting']
    HF_TOKEN: str = Path(HF_TOKEN_FNAME).read_text().strip()
    dir_path = home_dir = os.getenv("HOME", str(Path.home()))
    logger.info(f"Home directory: {dir_path}")
    
    # print the ec2 instance details for it to get logged
    logger.info(f"EC2 instance type: {ec2_metadata.instance_type}, AMI ID: {ec2_metadata.ami_id}")
    
    model_directory = _set_up(model_name, dir_path) 
    
    # if we have to load multiple copies of the model then we use the docker compose approach
    # otherwise we create a single directory with serving.properties and deployment script
    model_copies = experiment_config['inference_spec'].get('model_copies', '1')
    # multiple model copies only for neuron for now and only for DJL
    neuron_instance = experiment_config['instance_type'].startswith('inf2.') or experiment_config['instance_type'].startswith('trn1.')
    if neuron_instance is False or container_type != constants.CONTAINER_TYPE_DJL:
        model_copies = "1"
        logger.info(f"neuron_instance={neuron_instance}, container_type={container_type}, model_copies > 1 not supported, "
                    f"going to use model_copies={model_copies}")
    else:
        logger.info(f"neuron_instance={neuron_instance}, container_type={container_type}, model_copies={model_copies}")

    # if model_copies ==  "1":
    #     if serving_properties:
    #         file_path = os.path.join(model_directory, "serving.properties")
    #         Path(file_path).write_text(serving_properties)
    #         logger.info(f"The serving.properties file has been created in {file_path}")
    #     else:
    #         logger.info(f"no serving.properties content specified, not creating the file")
    # else:
    logger.info(f"model_copies={model_copies}, going to use the docker compose approach")
    if env is None:
        env = {"HF_TOKEN": HF_TOKEN}
    else:
        env["HF_TOKEN"] = HF_TOKEN
    logger.info(f"env={env}")
    prepare_for_neuron(model_id=Path(model_id).name,
                        num_model_copies=model_copies,
                        tp_degree=experiment_config['inference_spec']['tp_degree'],
                        image=image_uri,
                        user="djl",
                        shm_size=experiment_config['inference_spec']['shm_size'],
                        env=env,
                        serving_properties=serving_properties,
                        dir_path=dir_path)

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
                                                       env,
                                                       model_copies)

    logger.info("Running the deployment script")
    ran_container = _run_container(deployment_script_path)

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
