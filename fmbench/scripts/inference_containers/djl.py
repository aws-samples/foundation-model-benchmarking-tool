"""
DJL specific code
"""
import os
from pathlib import Path
import fmbench.scripts.constants as constants
from typing import Dict, List, Optional, Tuple, Union
from fmbench.scripts.inference_containers.utils import (STOP_AND_RM_CONTAINER,
                                                        FMBENCH_MODEL_CONTAINER_NAME)

CONFIG_PROPERTIES: str = """
            inference_address=http://0.0.0.0:{port}
            management_address=http://0.0.0.0:{port}
            cluster_address=http://0.0.0.0:8888
            model_store=/opt/ml/model
            load_models=ALL
            """

def create_script(region, image_uri, model_id, model_name, env_str, privileged_str, hf_token, directory):
    """
    Script for running the docker container for the inference server
    """
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

def create_djl_service(model_id: str, 
                        num_model_copies: int, 
                        devices_per_model: int, 
                        image: str, 
                        user: str, 
                        shm_size: str, 
                        env: List, 
                        base_port: int, 
                        accelerator: constants.ACCELERATOR_TYPE) -> Tuple:
    """
    Creates the service for DJL, setting up devices and other configurations.
    """
    try:
        cnames: List = []
        services: Dict = {}
        per_container_info_list: List = []
        home = str(Path.home())
        dir_path_on_host: str = os.path.join(home, Path(model_id).name)
        
        # Iterate through the number of model copies and prepare the docker service for djl
        for i in range(num_model_copies):
            cname = f"fmbench_model_container_{i+1}"
            cnames.append(cname)

            device_offset = devices_per_model * i

            if accelerator == constants.ACCELERATOR_TYPE.NEURON:
                devices = [f"/dev/neuron{j + device_offset}:/dev/neuron{j}" for j in range(devices_per_model)]
                extra_env = []
            else:
                devices = None
                gpus = ",".join([str(j + device_offset) for j in range(devices_per_model)])
                extra_env = [f"NVIDIA_VISIBLE_DEVICES={gpus}"]
                env = env + extra_env

            volumes = [f"{dir_path_on_host}/i{i+1}:/opt/ml/model:ro",
                       f"{dir_path_on_host}/i{i+1}/conf:/opt/djl/conf:ro",
                       f"{dir_path_on_host}/i{i+1}/model_server_logs:/opt/djl/logs"]
            # compute the port
            port = base_port + i

            service = {
                cname: {
                    "image": image,
                    "container_name": cname,
                    "user": user,
                    "shm_size": shm_size,
                    "devices": devices,
                    "environment": env,
                    "volumes": volumes,
                    "ports": [f"{port}:{port}"],
                    "deploy": {"restart_policy": {"condition": "on-failure"}}
                }
            }
            if accelerator == constants.ACCELERATOR_TYPE.NVIDIA:
                service[cname]['runtime'] = constants.ACCELERATOR_TYPE.NVIDIA.value
                service[cname].pop("devices")
            services.update(service)
            config_properties = CONFIG_PROPERTIES.format(port=port)


            nginx_server_lines = [f"        server {cname}:{port};"]
    
            per_container_info_list.append(dict(dir_path_on_host=f"{dir_path_on_host}/i{i+1}",
                                                config_properties=config_properties,
                                                container_name=cname,
                                                nginx_server_lines=nginx_server_lines))
    except Exception as e:
        logger.error(f"Error occurred while generating configuration files for djl/vllm: {e}")
        services, per_container_info_list=None, None
    # since the djl container comes up immediately and the nginx lb does not have to wait
    # so the nginx command is a noop
    nginx_command = None
    return services, per_container_info_list, nginx_command
