# script that prepares a directory structure to run multiple copies of a model
# using the DJL container. This code is written for Neuron for now but would be modified
# to run on GPUs as well and then maybe other inference containers as well.
# It creates a docker_compose.yml file that creates multiple containers and a load balancer.
# which provides a single endpoint to external applications that want to use these containers.

import os
import json
import yaml
import shutil
import docker
import logging
import subprocess
from os import listdir
from pathlib import Path
from os.path import isfile, join
import fmbench.scripts.constants as constants
from typing import Dict, List, Optional, Tuple, Union
import fmbench.scripts.inference_containers.djl as djl
import fmbench.scripts.inference_containers.vllm as vllm
import fmbench.scripts.inference_containers.triton as triton
import fmbench.scripts.inference_containers.utils as ic_utils

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def _create_config_files(model_id: str, 
                        num_model_copies: int, 
                        devices_per_model: int, 
                        image: str, 
                        tp_degree: int,
                        batch_size: int,
                        user: str, 
                        shm_size: str, 
                        env: List, 
                        accelerator: constants.ACCELERATOR_TYPE,
                        dir_path: str,
                        listen_port: int, 
                        backend: constants.BACKEND) -> Tuple:
    """
    Creates the docker compose yml file, nginx config file, these are common to all containers
    and then creates individual config.properties files

    version: '3.8'

    services:
      fmbench_model_container_1:
        image: 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.29.0-neuronx-sdk2.19.1
        # deepjavalibrary/djl-serving:0.29.0-pytorch-inf2 #
        container_name: fmbench_model_container_1
        user: djl
        shm_size: 12GB
        devices:
        - "/dev/neuron0:/dev/neuron0"
        - "/dev/neuron1:/dev/neuron1"
        - "/dev/neuron2:/dev/neuron2"
        - "/dev/neuron3:/dev/neuron3"
        - "/dev/neuron4:/dev/neuron4"
        - "/dev/neuron5:/dev/neuron5"
        environment:
        - MODEL_LOADING_TIMEOUT=2400
        - HF_TOKEN=<hf_token>
        volumes:
        - /home/ubuntu/Mistral-7B-Instruct-v0.2-i1:/opt/ml/model:ro
        - /home/ubuntu/Mistral-7B-Instruct-v0.2-i1/conf:/opt/djl/conf:ro
        - /home/ubuntu/Mistral-7B-Instruct-v0.2-i1/model_server_logs:/opt/djl/logs
        ports:
        - "8081:8081"
        deploy:
        restart_policy:
            condition: on-failure

      fmbench_model_container_2:
        image: 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.29.0-neuronx-sdk2.19.1    
        container_name: fmbench_model_container_2
        user: djl
        shm_size: 12GB
        devices:
        - "/dev/neuron6:/dev/neuron0"
        - "/dev/neuron7:/dev/neuron1"
        - "/dev/neuron8:/dev/neuron2"
        - "/dev/neuron9:/dev/neuron3"
        - "/dev/neuron10:/dev/neuron4"
        - "/dev/neuron11:/dev/neuron5"
        environment:
        - MODEL_LOADING_TIMEOUT=2400
        - HF_TOKEN=<hf_token>
        volumes:
        - /home/ubuntu/Mistral-7B-Instruct-v0.2-i2:/opt/ml/model:ro
        - /home/ubuntu/Mistral-7B-Instruct-v0.2-i2/conf:/opt/djl/conf:ro
        - /home/ubuntu/Mistral-7B-Instruct-v0.2-i2/model_server_logs:/opt/djl/logs
        ports:
        - "8082:8082"
        deploy:
        restart_policy:
            condition: on-failure

      loadbalancer:
        image: nginx:alpine
        container_name: fmbench_model_container_loadbalancer
        ports:
        - "8080:80"
        volumes:
        - ./nginx.conf:/etc/nginx/nginx.conf:ro
        depends_on:
        - fmbench_model_container_1
        - fmbench_model_container_2
        deploy:
        placement:
            constraints: [node.role == manager]
    """

    # do we need to create a load balancer?
    # if the model copies or the model containers are > 1
    # then yes..note that we could have a situation where multiple
    # model servers are created within a single container so go with
    # model copies as the guide rather than containers
    lb_needed = True if num_model_copies > 1 else False
    try:
        if lb_needed is False:
            base_port = listen_port
            logger.info(f"lb_needed={lb_needed}, going to make the container listen on {listen_port} instead of "
                        f"{base_port}")
        else:
            base_port = constants.BASE_PORT_FOR_CONTAINERS
        if user == constants.CONTAINER_TYPE_TRITON:            
            logger.info(f"container type is {constants.CONTAINER_TYPE_TRITON}, accelerator={accelerator}. "
                        f"preparing the service for docker-compose")
            services, per_container_info_list, nginx_command = triton.create_triton_service(model_id, 
                                                                    num_model_copies, 
                                                                    devices_per_model, 
                                                                    image, 
                                                                    user, 
                                                                    shm_size,
                                                                    env, 
                                                                    base_port,
                                                                    accelerator,
                                                                    tp_degree,
                                                                    batch_size, 
                                                                    backend)
        else:
            logger.info(f"Container type is {user}. Preparing the service for docker-compose")
            services, per_container_info_list, nginx_command = djl.create_djl_service(model_id, 
                                                                    num_model_copies, 
                                                                    devices_per_model, 
                                                                    image, 
                                                                    user, 
                                                                    shm_size, 
                                                                    env, 
                                                                    base_port,
                                                                    accelerator)
        # Load balancer setup
        if lb_needed is True:
            fmbench_container_names = [ci['container_name'] for ci in per_container_info_list]
            # if this is a triton container and 
            lb = {
                "image": "nginx:alpine",
                "container_name": "fmbench_model_container_load_balancer",
                "command": nginx_command,
                "restart": "on-failure",
                "ports": [f"{listen_port}:80"],
                "volumes": ["./nginx.conf:/etc/nginx/nginx.conf:ro"],
                "depends_on": fmbench_container_names,
                "deploy": {"placement": {"constraints": ['node.role == manager']}}
            }
            if nginx_command is None:
                lb.pop("command")
            services.update({"loadbalancer": lb})
        else:
            logger.info(f"not creating load balancer")
        docker_compose = {"services": services}

        # nginx.conf generation
        if lb_needed is True:
            # the container info list contains the nginx.conf "server fmbench_model_Container_1:8000" type lines
            # as well ..but..not all inference containers are the same..for DJL we would have one model server
            # per container (or even if we have multiple they listen on the same port (?) so we dont have to care
            # about it in the nginx ln) but in Triton the model servers listen on different ports and so even if
            # there is a single triton container it could have say 4 model servers on different ports..all this 
            # to say that the nginx_conf_lines is a list and so we have a list of list situation happening here and 
            # we need to flatten this list of list
            
            nginx_server_lines = [l for ci in per_container_info_list for l in ci['nginx_server_lines']]
            # make it into a string
            nginx_server_lines = "\n".join(nginx_server_lines)

            nginx_config = f"""
### Nginx Load Balancer
events {{}}
http {{
    upstream fmcluster {{
{nginx_server_lines}
    }}
    server {{
        listen 80;
        location / {{
            proxy_pass http://fmcluster;
        }}
    }}
}}
"""
        else:
            nginx_config = None
    except Exception as e:
        logger.error(f"Error occurred while generating config files: {e}")
        docker_compose, nginx_config, per_container_info_list = None, None, None
    return docker_compose, nginx_config, per_container_info_list


def prepare_docker_compose_yml(model_id: str,
                               model_copies: str,
                               inference_params: Dict,
                               image: str,
                               user: str,
                               shm_size: str,
                               env: Dict,
                               serving_properties: Optional[str],
                               dir_path: str,
                               listen_port: int) -> int:

    # convert the env dict to a list of k=v pairs
    env_as_list = []
    if env is not None:
        for k,v in env.items():
            env_as_list.append(f"{k}={v}")

    # this is specific to the triton container
    # then default it to None and let the inference
    # container the best default values for parameters
    container_params: Optional[Dict] = inference_params.get('container_params', dict())
    logger.info(f"container_params that will be used for {image} image: {container_params}")  

    # the following code handles backwards compatibility where batch_Size and tp_degree are now specified
    # as part of container params but if not found then it falls back to the original scheme where
    # they were specified as part of the inference params
    batch_size: int = container_params.get('batch_size', inference_params.get('batch_size', 4))
    tp_degree: int = container_params.get('tp_degree', inference_params.get('tp_degree', 1))

    backend: Optional[str] = inference_params.get('backend')
    triton_content: Optional[str] = None
    if backend is not None:
        if backend == constants.BACKEND.VLLM_BACKEND:
            triton_content = constants.TRITON_CONTENT_DIR_NAME_VLLM
        elif backend == constants.BACKEND.DJL_BACKEND:
            triton_content = constants.TRITON_CONTENT_DIR_NAME_DJL
        else:
            logger.error(f"No backend={backend} provided")
    else:
        logger.error(f"No backend={backend} provided")

    # first check if this is an NVIDIA instance or a AWS Chips instance    
    accelerator = ic_utils.get_accelerator_type()

    if accelerator == constants.ACCELERATOR_TYPE.NVIDIA:
        model_copies_as_int, devices_per_model = ic_utils.get_model_copies_to_start_nvidia(tp_degree, model_copies)
    else:
        model_copies_as_int, devices_per_model = ic_utils.get_model_copies_to_start_neuron(tp_degree, model_copies)
    
    # special handling for Triton && Neuron
    if user == constants.CONTAINER_TYPE_TRITON and accelerator == constants.ACCELERATOR_TYPE.NEURON:
        logger.info(f"running Triton container on neuron, creating specific config files for this combination")
        current_dir: str = os.path.dirname(os.path.realpath(__file__))
        triton_content: str = os.path.join(current_dir, triton_content)
        # handles custom tensor pd, batch size into the model repository files
        triton.handle_triton_serving_properties_and_inf_params(triton_content, 
                                                               tp_degree, 
                                                               container_params,
                                                               model_id,
                                                               backend)

    # create the docker compose and nginx.conf file in the top level
    # directory path for this model
    dir_path = os.path.join(dir_path, Path(model_id).name)
    os.makedirs(dir_path, exist_ok=True)
    docker_compose, nginx_config, per_container_info_list = _create_config_files(model_id,
                                                                                 model_copies_as_int,
                                                                                 devices_per_model,
                                                                                 image,
                                                                                 tp_degree,
                                                                                 batch_size,
                                                                                 user,
                                                                                 shm_size,
                                                                                 env_as_list,
                                                                                 accelerator,
                                                                                 dir_path,
                                                                                 listen_port, 
                                                                                 backend)

    yaml.Dumper.ignore_aliases = lambda self, data: True
    docker_compose_yaml = yaml.dump(docker_compose)


    # ready to create the common files: docker_compose.yml and nginx.conf
    # any inference server specific files are created by the _create_config_files function
    # which in turn calls inference server specific functions
    dc_path = os.path.join(dir_path, "docker-compose.yml")
    logger.info(f"writing docker compose to {dc_path}, contents --->\n{docker_compose_yaml}")    
    Path(dc_path).write_text(docker_compose_yaml)

    # create nginx.conf file
    if nginx_config is not None:
        ngc_path = os.path.join(dir_path, "nginx.conf")
        logger.info(f"writing nginx conf to {ngc_path}, contents --->\n{nginx_config}")    
        Path(ngc_path).write_text(nginx_config)
    else:
        logger.info(f"model_copies={model_copies}, nginx_config={nginx_config}, "
                    f"not creating nginx.conf")

    # create sub directories for each model instance
    for idx, pc in enumerate(per_container_info_list):
        logger.info(f"creating files for container {idx+1} of {len(per_container_info_list)}...")
        # create serving.properties
        if serving_properties is not None:
            os.makedirs(pc["dir_path_on_host"], exist_ok=True)
            sp_fpath = os.path.join(pc["dir_path_on_host"], "serving.properties")
            logger.info(f"writing {serving_properties} to {sp_fpath}")
            Path(sp_fpath).write_text(serving_properties)
        else:
            logger.info(f"serving_properties={serving_properties}, skipping")

        # write config.properties, only applicable to DJL
        if pc['config_properties'] is not None:
            conf_dir = os.path.join(pc["dir_path_on_host"], "conf")
            os.makedirs(conf_dir, exist_ok=True)
            cp_fpath = os.path.join(conf_dir, "config.properties")
            logger.info(f"writing {pc['config_properties']} to {cp_fpath}")
            Path(cp_fpath).write_text(pc['config_properties'])
        else:
            logger.info(f"config_properties={pc['config_properties']}, skipping")

    return model_copies_as_int
