"""
Deploys a model from HuggingFace on Amazon EC2

1. Configuration is read from the configured serving.properties file.
2. A hf_token.txt file is required to download the model from Hugging Face.
"""
# Import necessary libraries
import os
import re
import sys
import time
import json
import stat
import docker
import inspect
import logging
import requests
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Union
from fmbench.scripts import constants
from ec2_metadata import ec2_metadata
from fmbench.scripts.inference_containers import (djl, vllm, triton)
from fmbench.scripts.constants import (IS_NEURON_INSTANCE, LISTEN_PORT)
from fmbench.scripts.prepare_for_multi_model_containers import prepare_docker_compose_yml

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the platform where this script deploys the model
PLATFORM: str = constants.PLATFORM_EC2

# globals
HF_TOKEN_FNAME: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hf_token.txt")
FMBENCH_MODEL_CONTAINER_NAME: str = "fmbench_model_container"

def _set_up(model_name: str, local_model_path: str):
    """
    Create the model serving.properties file locally in a model directory
    """
    # make the model directory with serving.properties
    directory = os.path.join(local_model_path, model_name)
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Creating the local directory: {directory}")
    
    # return the directory we created
    return directory

def _create_deployment_script(image_uri,
                              container_type,
                              privileged_mode,
                              region,
                              model_name,
                              model_id,
                              hf_token,
                              directory,
                              model_loading_timeout,
                              env,
                              model_copies,
                              is_neuron_instance):
    """
    Write a deployment script for model container
    """
    logger.info(inspect.getargvalues(inspect.currentframe()))
    privileged_str: str = "--privileged" if privileged_mode else ""
    env_str: str = ""
    if env is not None:
        for k, v in env.items():
            env_str += f"-e {k}={v} "

    match container_type:
        case constants.CONTAINER_TYPE_DJL:
            deploy_script_content = djl.create_script(region, image_uri, model_id, model_name, env_str, privileged_str, hf_token, directory)
        case constants.CONTAINER_TYPE_VLLM:
            deploy_script_content = vllm.create_script(region, image_uri, model_id, model_name, env_str, privileged_str, hf_token, directory)
        case constants.CONTAINER_TYPE_TRITON:
            deploy_script_content = triton.create_script(region, image_uri, model_id, model_name, env_str, privileged_str, hf_token, directory)
        case _:
            raise ValueError(f"dont know how to handle container_type={container_type}")
    script_file_path = os.path.join(directory, "deploy_model.sh")
    Path(script_file_path).write_text(deploy_script_content)
    st = os.stat(script_file_path)
    os.chmod(script_file_path, st.st_mode | stat.S_IEXEC)

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
        logger.info(f"going to run script {script_file_path}")
        subprocess.run(["bash", script_file_path], check=True)
        logger.info(f"done running bash script")
        return True    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return False

def _check_model_deployment(endpoint, model_id, container_type, model_loading_timeout, model_copies, is_neuron_instance):
    """
    Check the model deployment status and wait for the model to be ready.
    """
    start_time = time.time()
    logger.info(f"Checking deployment status at {endpoint}")
    if container_type == constants.CONTAINER_TYPE_DJL:
        data = {"inputs": "tell me a story of the little red riding hood"}
    elif container_type == constants.CONTAINER_TYPE_VLLM:
        data = {"model": model_id,  # Specify the model to use
                "prompt": "tell me a story of the little red riding hood",}
    elif container_type == constants.CONTAINER_TYPE_TRITON:
        if is_neuron_instance:
            data = {
                    "text_input": "tell me a story of the little red riding hood",
                    "sampling_parameters": json.dumps({"top_k": 50})
                }
        else:
            data = {"text_input": "tell me a story of the little red riding hood", "max_tokens": 50}
    headers = {"content-type": "application/json"}
    container_check_timeout = 60
    logger.info(f"going to check every {container_check_timeout}s for the inference endpoint to be up...")
    logger.info(f"this will take at least 10 minutes or so, please be patient...")
    while time.time() - start_time < model_loading_timeout:        
        try:
            response = requests.post(endpoint, headers=headers, json=data)
            logger.info(f"response is: {response.text}")
            if response.status_code == 200:
                logger.info("model deployment successful!")

                # if model_copies != 1 then wait for some more time to give
                # all the containers a chance to be up
                if model_copies > 1:
                    # this extra wait is not needed if this is a triton server running on a gpu
                    # because we start multiple model servers in a single container
                    if is_neuron_instance is False and container_type == constants.CONTAINER_TYPE_TRITON:
                        logger.info(f"this is a triton container running on a gpu instance, skipping the additional wait")
                    else:
                        additional_sleep_time = model_copies * container_check_timeout
                        logger.info(f"since model_copies={model_copies}, waiting for an addition {additional_sleep_time}s "
                                    f"to allow all model endpoints to come up")
                        time.sleep(additional_sleep_time)
                total_wait_time = time.time() - start_time
                logger.info("marking endpoint in service, total_wait_time={total_wait_time}s")
                return "InService"
            else:
                logger.info(f"model deployment is not ready yet. Return code: {response.status_code}")
                logger.info(f"waiting for {container_check_timeout}s before retrying")
                time.sleep(container_check_timeout)
        except Exception as e:
            logger.error(f"error occurred while deploying the endpoint: {e}")
            logger.info(f"waiting for {container_check_timeout}s before retrying")
            time.sleep(container_check_timeout)
    return "Failed"

def deploy(experiment_config: Dict, role_arn: str) -> Dict:
    """
    Function to deploy the model and create the endpoint
    """
    image_uri: str = experiment_config['image_uri']
    ep_name: str = experiment_config['ep_name']
    model_id: str = experiment_config['model_id']
    model_name: str = Path(model_id).name
    logger.info(f"Going to deploy model: {model_name}")

    ep_name: str = experiment_config['ep_name']
    # extract port number from the endpoint name
    # 8080 from "http://localhost:8080/v2/models/ensemble/generate"
    # this is the port on which either the inference container (if there
    # is a single container) or the load balancer in case of multiple containers
    # will listen for new connnections
    match = re.search(r"http://[^:/]+:(\d+)", ep_name)
    listen_port = int(match.group(1)) if match else constants.LISTEN_PORT
    logger.info(f"ep_name={ep_name}, listen_port={listen_port}")
    region: str = experiment_config['region']
    privileged_mode: str = experiment_config['ec2'].get('privileged_mode', False)
    container_type: str = experiment_config['inference_spec'].get('container_type', constants.CONTAINER_TYPE_DJL)
    env = experiment_config.get('env')
    container_params: str = experiment_config['inference_spec'].get('container_params', None)
    if container_params is None:
        serving_properties = experiment_config.get('serving.properties', None)
        logger.info(f"Serving properties found in experiment config: {serving_properties}")
    else: 
        serving_properties = container_params.get('serving.properties', None)
        logger.info(f"Serving properties found in container params in the config: {serving_properties}")
    logger.info(f"Serving properties provided to deploy on container type={container_type} using image_uri={image_uri}: {serving_properties}")
    model_loading_timeout: int = experiment_config['ec2']['model_loading_timeout']
    
    HF_TOKEN: str = Path(HF_TOKEN_FNAME).read_text().strip()
    dir_path = home_dir = os.getenv("HOME", str(Path.home()))
    logger.info(f"Home directory: {dir_path}")
    
    # print the ec2 instance details for it to get logged
    logger.info(f"EC2 instance type: {ec2_metadata.instance_type}, AMI ID: {ec2_metadata.ami_id}")
    
    # HF token is mandatory even if no env vars were specific in the config file    
    if env is None:
        env = {"HF_TOKEN": HF_TOKEN}
    else:
        env["HF_TOKEN"] = HF_TOKEN
    logger.info(f"env={env}")    

    model_directory = _set_up(model_name, dir_path)
    is_neuron_instance = IS_NEURON_INSTANCE(experiment_config['instance_type'])
    model_copies = experiment_config['inference_spec'].get('model_copies', '1')


    # if this is a neuron instance and we are using the djl serving inference container
    # then use the docker compose approach so we first create the docker compose file
    # and then create the deployment script, otherwise we create the deployment script
    # directly (see call to _create_deployment_script below)
    model_copies_actual = 1
    if (container_type == constants.CONTAINER_TYPE_DJL) or (container_type == constants.CONTAINER_TYPE_TRITON):  
        logger.info(f"container_type={container_type}, is_neuron_instance={is_neuron_instance}, going to create docker compose yml")
        model_copies_actual = prepare_docker_compose_yml(model_id=model_id,
                                                         model_copies=model_copies,
                                                         inference_params=experiment_config['inference_spec'],
                                                         image=image_uri,
                                                         user=container_type,
                                                         shm_size=experiment_config['inference_spec']['shm_size'],
                                                         env=env,
                                                         serving_properties=serving_properties,
                                                         dir_path=dir_path,
                                                         listen_port=listen_port)   
    logger.info("Creating the deployment script in model directory")
    deployment_script_path = _create_deployment_script(image_uri,
                                                       container_type,
                                                       privileged_mode,
                                                       region,
                                                       model_name,
                                                       model_id,
                                                       HF_TOKEN,
                                                       model_directory,
                                                       model_loading_timeout,
                                                       env,
                                                       model_copies,
                                                       is_neuron_instance)

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
        ep_status = _check_model_deployment(ep_name, model_id, container_type, model_loading_timeout, model_copies_actual, is_neuron_instance)
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
