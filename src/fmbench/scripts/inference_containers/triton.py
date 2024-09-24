"""
Triton specific code
"""
import os
import json
import stat
import shutil
import logging
from pathlib import Path
from typing import Tuple, List
import fmbench.scripts.constants as constants
from typing import Dict, List, Optional, Tuple, Union
from fmbench.scripts.inference_containers.utils import (STOP_AND_RM_CONTAINER,
                                                        FMBENCH_MODEL_CONTAINER_NAME)

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # using the locally built triton image

        # Run the new Docker container with specified settings
        cd {directory}
        # shutdown existing docker compose        
        docker compose down
        {STOP_AND_RM_CONTAINER}

        # download the model
        export HF_TOKEN={hf_token}
        pip freeze | grep huggingface-hub
        RESULT=$?
        if [ $RESULT -eq 0 ]; then
          echo huggingface-hub is already installed..
        else
          echo huggingface-hub is not installed, going to install it now
          pip install -U "huggingface_hub[cli]"
        fi

        huggingface-cli download {model_id} --local-dir $HOME/{model_id}

        # bring up the inference container and load balancer
        docker compose up -d
        cd -
        echo "started docker compose in daemon mode"
    """
    return script

def handle_triton_serving_properties_and_inf_params(triton_dir: str,
                                                    tp_degree: int,
                                                    batch_size: int,
                                                    model_json_params: Dict,
                                                    hf_model_id: str):
    """
    Substitutes parameters within the triton model repository files for the vllm backend: config.pbtxt, model.json and substitutes the batch size, 
    hf mdoel id from the configuration file into those files before the model repository is prepared within the container.
    """
    try:
        # iterate through each of the file in the triton directory
        # and substitute the HF model id, TP degree and batch size in 
        # model.json and config.pbtxt. model.py remains the same for 
        # all HF models
        for root, dirs, files in os.walk(triton_dir):
            for file in files:
                file_path = os.path.join(root, file)

                if file == "model.json":
                    with open(file_path, "r") as f:
                        content = json.load(f)
                    # Replace placeholders in model.json
                    # this includes TP degree, batch size, 
                    # and HF model id
                    content["tensor_parallel_size"] = tp_degree
                    content['model'] = hf_model_id
                    # update the model.json to contain additional variables, such as
                    # max_num_seqs, max_model_len, batch_size and more
                    content.update(model_json_params)
                    with open(file_path, "w") as f:
                        json.dump(content, f, indent=2)
                    logger.info(f"Updated {file_path} with tp_degree={tp_degree}, model_json_params={model_json_params}")
                    
                # upate the config.pbtxt with the batch size parameter fetched from the configuration file
                elif file == "config.pbtxt":
                    with open(file_path, "r") as f:
                        content = f.read()
                    content = content.replace("{BATCH_SIZE}", str(batch_size))
                    with open(file_path, "w") as f:
                        f.write(content)
                    logger.info(f"Updated {file_path} with batch_size={batch_size}.")
                else:
                    # If there are files within the triton folder that do not need
                    # to have values subsituted within it, then call it out.
                    logger.info(f"No substitutions needed for {file_path}")
    except Exception as e:
        raise Exception(f"Error occurred while preparing files for the triton model container with vllm backend: {e}")


def _create_triton_service_neuron(model_id: str, 
                          num_model_copies: int, 
                          devices_per_model: int, 
                          image: str,
                          user: str, 
                          shm_size: str, 
                          env: List,
                          base_port: int, 
                          accelerator: constants.ACCELERATOR_TYPE) -> Tuple:
    """
    Creates the Triton service-specific part of the docker compose file. This function is responsible for handling the volume
    mounting, and other aspects to the docker compose file, such as the entrypoint command, port mapping, and more.
    """
    try:
        # initialize the cnames, services dictionary for triton on neuron, 
        # and more 
        cnames: List = []
        services: Dict = {}
        per_container_info_list: List = []
        ports_per_model_server: int = 3 # http, grps, metrics
        home = str(Path.home())
        dir_path_on_host: str = os.path.join(home, Path(model_id).name)
        
        # Iterate through the number of model copies and prepare the docker service for triton
        for i in range(num_model_copies):
            cname: str = f"fmbench_model_container_{i+1}"
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

            # Setup Triton model content for each instance
            instance_dir: str = os.path.join(dir_path_on_host, f"i{i+1}")
            triton_instance_dir: str = os.path.join(instance_dir, "triton")
            os.makedirs(triton_instance_dir, exist_ok=True)
            current_dir: str = os.path.dirname(os.path.realpath(__file__))
            parent_dir: str = os.path.abspath(os.path.join(current_dir, os.pardir))
            triton_content: str = os.path.join(parent_dir, constants.TRITON_CONTENT_DIR_NAME)

            # Copy all files from triton_content to the instance's triton directory
            for item in os.listdir(triton_content):
                s = os.path.join(triton_content, item)
                d = os.path.join(triton_instance_dir, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
                elif os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
            # Set execute permissions for the script. The triton volumes contain the model repostory that
            # are mapped into the container and used during deployment
            os.chmod(os.path.join(triton_instance_dir, os.path.basename(constants.TRITON_INFERENCE_SCRIPT)), 0o755)
            volumes = [f"{triton_instance_dir}:/scripts/triton:rw",
                       f"{triton_instance_dir}:/triton:rw"]
            # Create the Triton service
            total_neuron_cores = num_model_copies * devices_per_model * 2
            service = {
                cname: {
                    "image": image,
                    "container_name": cname,
                    "shm_size": shm_size,
                    "devices": devices,
                    "environment": env,
                    "volumes": volumes,
                    "ports": [f"{base_port + i*ports_per_model_server}:{base_port + i*ports_per_model_server}"],
                    "deploy": {"restart_policy": {"condition": "on-failure"}},
                    "command": f"{constants.TRITON_INFERENCE_SCRIPT} {model_id} {Path(model_id).name} {base_port + i*ports_per_model_server} {total_neuron_cores}"
                }
            }
            services.update(service)
            # for the triton container we could have multiple model servers within the same container
            nginx_server_lines = [f"        server {cname}:{base_port + i*ports_per_model_server};"]
            config_properties = CONFIG_PROPERTIES.format(port=(base_port + i*ports_per_model_server))
            per_container_info_list.append(dict(dir_path_on_host=instance_dir, 
                                                config_properties=config_properties, 
                                                container_name=cname,
                                                nginx_server_lines=nginx_server_lines))
    except Exception as e:
        logger.error(f"Error occurred while creating configuration files for triton: {e}")
        services, per_container_info_list = None, None
    nginx_command = "sh -c \"echo going to sleep for 240s && sleep 240 && echo after sleep && nginx -g \'daemon off;\' && echo started nginx\""
    return services, per_container_info_list, nginx_command


def _create_triton_service_gpu(model_id: str, 
                          num_model_copies: int, 
                          devices_per_model: int, 
                          image: str, 
                          user: str, 
                          shm_size: str, 
                          env: List,
                          base_port: int, 
                          accelerator: constants.ACCELERATOR_TYPE,
                          tp_degree: int,
                          batch_size: int) -> Tuple:
    """
    Creates the Triton service-specific part of the docker compose file. This function is responsible for handling the volume
    mounting, and other aspects to the docker compose file, such as the entrypoint command, port mapping, and more.
    """
    try:
        """
        the gpu version of services looks like this
        services:
    fmbench_model_container:
        image: nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3
        container_name: fmbench_model_container  # Name of the container
        runtime: nvidia  # Enables GPU support for the container
        shm_size: 12g  # Shared memory size
        ulimits:
          memlock: -1  # Remove memory locking limits
          stack: 67108864  # Set stack size
        ports:
        # sufficient for upto 4 instances of the model
        # add more if needed
        - 8000:8000
        - 8003:8003
        - 8006:8006
        - 8009:8009
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: all  # Use all available GPUs
                  capabilities: [gpu]
        volumes:
        - ${HOME}/tensorrtllm_backend:/tensorrtllm_backend
        - ${HOME}/${MODEL_ID}:/${MODEL_ID}
        - ${HOME}/engines:/engines
        - ${HOME}/deploy_on_triton/scripts:/scripts
        #network_mode: host  # Use the host's network stack
        tty: true  # Allocate a pseudo-TTY (interactive terminal)
        command: bash -c "/scripts/serve_model.sh ${MODEL_ID} ${TP_DEGREE} ${BATCH_SIZE} ${MODEL_COPIES} && bash"  # Run script and keep the container alive with bash
        restart: on-failure  # Ensure container restarts if it stops unexpectedly
        """
        cnames: List = []
        services: Dict = {}
        per_container_info_list: List = []
        home = str(Path.home())
        dir_path_on_host: str = os.path.join(home, Path(model_id).name)
        
        cname: str = FMBENCH_MODEL_CONTAINER_NAME
        ports_per_model_server: int = 3 # http, grps, metrics
        logger.info(f"_create_triton_service_gpu, model_id=\"{model_id}\", home=\"{home}\", dir_path_on_host=\"{dir_path_on_host}\", "
                    f"tp_degree=\"{tp_degree}\", batch_size=\"{batch_size}\", num_model_copies=\"{num_model_copies}\"")
        volumes = [f"{home}/tensorrtllm_backend:/tensorrtllm_backend",
                   f"{home}/{model_id}:/{model_id}",
                   f"{dir_path_on_host}/engines:/engines",
                   f"{dir_path_on_host}/scripts:/scripts"]
        
        # copy trtiton serving script
        triton_scripts_dir = os.path.join(dir_path_on_host, "scripts")
        os.makedirs(triton_scripts_dir, exist_ok=True)
        triton_serve_model_script_dst_path = os.path.join(triton_scripts_dir, constants.TRITON_SERVE_SCRIPT)
        script_dir_path = Path( __file__ ).parent.absolute()
        triton_serve_model_script_src_path = os.path.join(script_dir_path, constants.TRITON_SERVE_SCRIPT)
        logger.info(f"going to copy {triton_serve_model_script_src_path} to {triton_serve_model_script_dst_path}")
        shutil.copyfile(triton_serve_model_script_src_path, triton_serve_model_script_dst_path)
        st = os.stat(triton_serve_model_script_dst_path)
        os.chmod(triton_serve_model_script_dst_path, st.st_mode | stat.S_IEXEC)

        # setup the services section
        service = {
                cname: {
                    "image": image,
                    "container_name": cname,
                    "shm_size": shm_size,
                    "ulimits": {"memlock": -1, "stack": 67108864},
                    "volumes": volumes,
                    "ports": [f"{base_port + i*ports_per_model_server}:{base_port + i*ports_per_model_server}" for i in range(num_model_copies)],
                    "deploy": {"resources": {"reservations": {"devices": [{"driver": "nvidia", "count": "all", "capabilities": ['gpu']}]}}},
                    "tty": True,                    
                    "command": f"bash -c \"/scripts/{constants.TRITON_SERVE_SCRIPT} {model_id} {tp_degree} {batch_size} {num_model_copies} {base_port} && bash\"",
                    "restart": "on-failure"
                }
            }
        services.update(service)
        # for the triton container we could have multiple model servers within the same container
        nginx_server_lines = [f"        server {cname}:{base_port + i*ports_per_model_server};" for i in range(num_model_copies)]
        per_container_info_list.append(dict(dir_path_on_host=dir_path_on_host,
                                            config_properties=None,
                                            container_name=cname,
                                            nginx_server_lines=nginx_server_lines))
    except Exception as e:
        logger.error(f"Error occurred while creating configuration files for triton: {e}")
        services, per_container_info_list = None, None
    # ask the nginx lb to wait for a few minutes for the triton container to come up
    nginx_command = "sh -c \"echo going to sleep for 240s && sleep 240 && echo after sleep && nginx -g \'daemon off;\' && echo started nginx\""
    return services, per_container_info_list, nginx_command

def create_triton_service(model_id: str, 
                          num_model_copies: int, 
                          devices_per_model: int, 
                          image: str, 
                          user: str, 
                          shm_size: str, 
                          env: List,
                          base_port: int, 
                          accelerator: constants.ACCELERATOR_TYPE,
                          tp_degree: int,
                          batch_size: int) -> Tuple:
    """
    Creates the Triton service-specific part of the docker compose file. This function is responsible for handling the volume
    mounting, and other aspects to the docker compose file, such as the entrypoint command, port mapping, and more.
    """

    if accelerator == constants.ACCELERATOR_TYPE.NEURON:
        logger.info(f"accelerator={accelerator}, calling the neuron version of this function")
        return _create_triton_service_neuron(model_id, 
                          num_model_copies, 
                          devices_per_model, 
                          image, 
                          user, 
                          shm_size, 
                          env,
                          base_port, 
                          accelerator) 
    else:
        logger.info(f"accelerator={accelerator}, calling the gpu version of this function")
        return _create_triton_service_gpu(model_id, 
                          num_model_copies, 
                          devices_per_model, 
                          image, 
                          user, 
                          shm_size, 
                          env,
                          base_port, 
                          accelerator,
                          tp_degree,
                          batch_size) 


