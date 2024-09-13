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

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PROPERTIES: str = """
            inference_address=http://0.0.0.0:{port}
            management_address=http://0.0.0.0:{port}
            cluster_address=http://0.0.0.0:8888
            model_store=/opt/ml/model
            load_models=ALL
            """

def _handle_triton_serving_properties_and_inf_params(triton_dir: str, 
                                                     tp_degree: int, 
                                                     batch_size: int, 
                                                     n_positions: int,
                                                     on_device_embedding_parameters: dict,
                                                     max_new_tokens: int,
                                                     context_len: int,
                                                     inference_parameters: Dict, 
                                                     hf_model_id: str):
    """
    Substitutes parameters within the triton model repository files: config.pbtxt, model.json, model.py and substitutes the batch size, 
    tensor parallel degree, hf mdoel id, and n positions from the configuration file into those files before the model repository is prepared 
    within the container.
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
                    content["tp_degree"] = tp_degree
                    content["batch_size"] = batch_size
                    content['model'] = hf_model_id
                    content['tokenizer'] = hf_model_id
                    content['n_positions'] = n_positions
                    # If neuron config is in the model.json, then replace with what is given
                    # in the config file, else go with default. Default values are: 
                    # "on_device_embedding": {"max_length": 8192, "top_k": 50, "do_sample": true}
                    if 'neuron_config' in content:
                        content['neuron_config']['on_device_embedding'] = on_device_embedding_parameters
                        logger.info(f"Updated neuron_config['on_device_embedding'] with values from inference_parameters: {on_device_embedding_parameters}")
                    else:
                        logger.info("neuron_config not found in inference_spec; leaving on_device_embedding as is")
                    with open(file_path, "w") as f:
                        json.dump(content, f, indent=2)
                    logger.info(f"Updated {file_path} with tp_degree={tp_degree}, batch_size={batch_size}, and inference_parameters.")

                # upate the config.pbtxt with the batch size parameter fetched from the configuration file
                elif file == "config.pbtxt":
                    with open(file_path, "r") as f:
                        content = f.read()
                    content = content.replace("{BATCH_SIZE}", str(batch_size))
                    with open(file_path, "w") as f:
                        f.write(content)
                    logger.info(f"Updated {file_path} with batch_size={batch_size}.")
                
                # update the model.py file with the model context length and the max new tokens
                # that is passed within the model container
                elif file == "model.py":
                    with open(file_path, "r") as f:
                        content = f.read()
                    content = content.replace("{MODEL_MAX_LEN}", str(context_len))
                    content = content.replace("{MAX_NEW_TOKENS}", str(max_new_tokens))
                    with open(file_path, "w") as f:
                        f.write(content)
                        logger.info(f"Updated {file_path} with context_len={context_len} and max_new_tokens={max_new_tokens}.")
                else:
                    # If there are files within the triton folder that do not need
                    # to have values subsituted within it, then call it out.
                    logger.info(f"No substitutions needed for {file_path}")
    except Exception as e:
        raise Exception(f"Error occurred while preparing files for the triton model container: {e}")

                                                                       
def _create_triton_service(model_id: str, 
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
        cnames: List = []
        services: Dict = {}
        per_container_info_list: List = []
        dir_path_on_host: str = f"/home/ubuntu/{Path(model_id).name}"

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

            # Compute port
            port = base_port + i + 1

            # Setup Triton model content for each instance
            instance_dir: str = os.path.join(dir_path_on_host, f"i{i+1}")
            triton_instance_dir: str = os.path.join(instance_dir, "triton")
            os.makedirs(triton_instance_dir, exist_ok=True)
            current_dir = os.path.dirname(os.path.realpath(__file__))
            triton_content = os.path.join(current_dir, "triton")

            # Copy all files from triton_content to the instance's triton directory
            for item in os.listdir(triton_content):
                s = os.path.join(triton_content, item)
                d = os.path.join(triton_instance_dir, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
                elif os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
            # Set execute permissions for the script
            os.chmod(os.path.join(triton_instance_dir, os.path.basename(constants.TRITON_INFERENCE_SCRIPT)), 0o755)
            volumes = [f"{triton_instance_dir}:/scripts/triton:rw",
                       f"{triton_instance_dir}:/triton:rw"]
            # Create the Triton service
            total_neuron_cores = num_model_copies * devices_per_model * 2
            service = {
                cname: {
                    "image": image,
                    "container_name": cname,
                    "user": '',
                    "shm_size": shm_size,
                    "devices": devices,
                    "environment": env,
                    "volumes": volumes,
                    "ports": [f"{port}:{port}"],
                    "deploy": {"restart_policy": {"condition": "on-failure"}},
                    "command": f"{constants.TRITON_INFERENCE_SCRIPT} {model_id} {Path(model_id).name} {port} {total_neuron_cores}"
                }
            }
            services.update(service)
            config_properties = CONFIG_PROPERTIES.format(port=port)
            per_container_info_list.append(dict(dir_path_on_host=instance_dir, config_properties=config_properties))
    except Exception as e:
        logger.error(f"Error occurred while creating configuration files for triton: {e}")
        services, per_container_info_list=None, None
    return services, per_container_info_list

def _create_djl_service(model_id: str, 
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
        dir_path_on_host: str = f"/home/ubuntu/{Path(model_id).name}"

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
            port = base_port + i + 1

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
            per_container_info_list.append(dict(dir_path_on_host=f"{dir_path_on_host}/i{i+1}", config_properties=config_properties))
    except Exception as e:
        logger.error(f"Error occurred while generating configuration files for djl/vllm: {e}")
        services, per_container_info_list=None, None
    return services, per_container_info_list


def _create_config_files(model_id: str, 
                        num_model_copies: int, 
                        devices_per_model: int, 
                        image: str, 
                        tp_degree: int,
                        user: str, 
                        shm_size: str, 
                        env: List, 
                        accelerator: constants.ACCELERATOR_TYPE) -> Tuple:
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
    try:
        base_port = 8079 if num_model_copies == 1 else 8080
        if user == constants.CONTAINER_TYPE_TRITON:
            logger.info(f"Container type is {constants.CONTAINER_TYPE_TRITON}. Preparing the service for docker-compose")
            services, per_container_info_list = _create_triton_service(model_id, 
                                                                       num_model_copies, 
                                                                       devices_per_model, 
                                                                       image, 
                                                                       user, 
                                                                       shm_size, 
                                                                       env, 
                                                                       base_port,
                                                                       accelerator)
        else:
            logger.info(f"Container type is {user}. Preparing the service for docker-compose")
            services, per_container_info_list = _create_djl_service(model_id, 
                                                                    num_model_copies, 
                                                                    devices_per_model, 
                                                                    image, 
                                                                    user, 
                                                                    shm_size, 
                                                                    env, 
                                                                    base_port,
                                                                    accelerator)
        # Load balancer setup
        if num_model_copies > 1:
            lb = {
                "image": "nginx:alpine",
                "container_name": "fmbench_model_container_load_balancer",
                "ports": ["8080:80"],
                "volumes": ["./nginx.conf:/etc/nginx/nginx.conf:ro"],
                "depends_on": [f"fmbench_model_container_{i+1}" for i in range(num_model_copies)],
                "deploy": {"placement": {"constraints": ['node.role == manager']}}
            }
            services.update({"loadbalancer": lb})
        docker_compose = {"services": services}
        # nginx.conf generation
        if num_model_copies > 1:
            nginx_server_lines = "\n".join([f"        server fmbench_model_container_{i+1}:{base_port + i + 1};" for i in range(num_model_copies)])
            nginx_config = f"""
            ### Nginx Load Balancer
            events {{}}
            http {{
                upstream djlcluster {{
{nginx_server_lines}
                }}
                server {{
                    listen 80;
                    location / {{
                        proxy_pass http://djlcluster;
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

def _get_model_copies_to_start_neuron(tp_degree: int, model_copies: str) -> Tuple[int, int]:
    logger.info(f"_get_model_copies_to_start_neuron, tp_degree={tp_degree}, model_copies={model_copies}")
    # get the number of neuron cores
    cmd = ["neuron-ls -j"]
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True,
                               shell=True)
    std_out, std_err = process.communicate()
    logger.info(std_out.strip())
    logger.info(std_err)

    if std_err != '':
        logger.error("error determining neuron info, exiting")
        return None

    # convert output to a dict for easy parsing
    neuron_info = json.loads(std_out)
    num_neuron_devices = len(neuron_info)
    num_neuron_cores = 2 * num_neuron_devices

    if model_copies == constants.MODEL_COPIES.AUTO:
        model_copies_as_int = 1
        devices_per_model = num_neuron_devices
        logger.info(f"model copies is set to {model_copies}, model_copies_as_int={model_copies_as_int}, "
                    f"devices_per_model={devices_per_model}")
        return model_copies_as_int, devices_per_model
              
    # tensor parallelism requires as many neuron cores as tp degree
    # and the number of devices required is half of number of cores
    neuron_devices_needed_per_model_copy = int(tp_degree / 2)
    model_copies_possible = int(num_neuron_devices/neuron_devices_needed_per_model_copy)
    # if model_copies is max then load as many copies as possible
    if isinstance(model_copies, str):
        if model_copies == constants.MODEL_COPIES.MAX:
            model_copies = model_copies_possible
            logger.info(f"model_copies was set to \"{constants.MODEL_COPIES.MAX}\", "
                        f"this instance can support a max of {model_copies} model copies, "
                        f"going to load {model_copies} copies")
        else:            
            logger.info(f"model_copies set to a str value={model_copies}, "
                        f"will see if we can load these many models")
            model_copies = int(model_copies)
    else:
        logger.info(f"model_copies set to a numerical value={model_copies}, "
                    f"will see if we can load these many models")

    num_devices_needed = model_copies * neuron_devices_needed_per_model_copy
    
    logger.info(f"model_copies={model_copies}, tp_degree={tp_degree},\n"
                f"num_neuron_devices={num_neuron_devices}, num_neuron_cores={num_neuron_cores},\n"
                f"neuron_devices_needed_per_model_copy={neuron_devices_needed_per_model_copy}, num_devices_needed={num_devices_needed},\n"
                f"model_copies_possible={model_copies_possible}")

    if model_copies_possible < model_copies:
        logger.error(f"model_copies={model_copies} but model_copies_possible={model_copies_possible}, "
                     f"setting model_copies to max possible for this instance which is {model_copies_possible}")
        model_copies = model_copies_possible
    else:
        logger.error(f"model_copies={model_copies} and model_copies_possible={model_copies_possible}, "
                     f"it is possible to run {model_copies} models, going with that")

    return model_copies, neuron_devices_needed_per_model_copy

def _get_model_copies_to_start_nvidia(tp_degree: int, model_copies: str) -> Tuple[int, int]:
    logger.info(f"_get_model_copies_to_start_nvidia, tp_degree={tp_degree}, model_copies={model_copies}")
    # get the number of neuron cores
    cmd = ["nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"]
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True,
                               shell=True)
    std_out, std_err = process.communicate()
    logger.info(std_out.strip())
    logger.info(std_err)

    if std_err != '':
        logger.error("error determining neuron info, exiting")
        return None

    num_gpus = int(std_out.strip())

    if model_copies == constants.MODEL_COPIES.AUTO:
        model_copies_as_int = 1
        devices_per_model = num_gpus
        logger.info(f"model copies is set to {model_copies}, model_copies_as_int={model_copies_as_int}, "
                    f"devices_per_model={devices_per_model}")
        return model_copies_as_int, devices_per_model
    
    # tensor parallelism requires as many gpus as tp degree
    gpus_needed_per_model_copy = tp_degree
    model_copies_possible = int(num_gpus/gpus_needed_per_model_copy)
    # if model_copies is max then load as many copies as possible
    if isinstance(model_copies, str):
        if model_copies == constants.MODEL_COPIES.MAX:
            model_copies = model_copies_possible
            logger.info(f"model_copies was set to \"{constants.MODEL_COPIES.MAX}\", "
                        f"this instance can support a max of {model_copies} model copies, "
                        f"going to load {model_copies} copies")
        else:            
            logger.info(f"model_copies set to a str value={model_copies}, "
                        f"will see if we can load these many models")
            model_copies = int(model_copies)
    else:
        logger.info(f"model_copies set to a numerical value={model_copies}, "
                    f"will see if we can load these many models")

    num_gpus_needed = model_copies * gpus_needed_per_model_copy
    
    logger.info(f"model_copies={model_copies}, tp_degree={tp_degree},\n"
                f"num_gpus={num_gpus},\n"
                f"gpus_needed_per_model_copy={gpus_needed_per_model_copy}, num_gpus_needed={num_gpus_needed},\n"
                f"model_copies_possible={model_copies_possible}")

    if model_copies_possible < model_copies:
        logger.error(f"model_copies={model_copies} but model_copies_possible={model_copies_possible}, "
                     f"setting model_copies to max possible for this instance which is {model_copies_possible}")
        model_copies = model_copies_possible
    else:
        logger.error(f"model_copies={model_copies} and model_copies_possible={model_copies_possible}, "
                     f"it is possible to run {model_copies} models, going with that")

    return model_copies, gpus_needed_per_model_copy


def prepare_docker_compose_yml(model_name: str,
                               model_id: str,
                               model_copies: str,
                               inference_params: Dict,
                               image: str,
                               user: str,
                               shm_size: str,
                               env: Dict,
                               serving_properties: Optional[str],
                               dir_path: str) -> int:

    # convert the env dict to a list of k=v pairs
    env_as_list = []
    if env is not None:
        for k,v in env.items():
            env_as_list.append(f"{k}={v}")

    # Get the tp degree, batch size, n_positions and inference parameters that are used
    # during model deployment. This is representative of serving.properties for models served 
    # on the triton container
    tp_degree: int = inference_params.get('tp_degree', None)
    batch_size: int = inference_params.get('batch_size', 4)
    n_positions: int = inference_params.get('n_positions', 8192)
    # this is specific to the triton container
    max_new_tokens: int = inference_params.get('max_new_tokens', 100)
    context_len: int = inference_params.get('context_len', 8192)
    on_device_embedding_parameters: dict = inference_params.get('neuron_config', dict(max_length=8192, 
                                                                                   top_k=50, 
                                                                                   do_sample=True))
    inference_parameters: Dict = inference_params.get('parameters', None)

    if user == constants.CONTAINER_TYPE_TRITON:
        current_dir: str = os.path.dirname(os.path.realpath(__file__))
        triton_content: str = os.path.join(current_dir, "triton")
        # handles custom tensor pd, batch size into the model repository files
        _handle_triton_serving_properties_and_inf_params(triton_content, 
                                                        tp_degree, 
                                                        batch_size, 
                                                        n_positions, 
                                                        on_device_embedding_parameters,
                                                        max_new_tokens,
                                                        context_len,
                                                        inference_parameters, 
                                                        model_id)

    # first check if this is an NVIDIA instance or a AWS Chips instance
    # we do this by checking for nvidia-smi and neuron-ls, they should be
    # present on NVIDIA and AWS Chips instances respectively
    NVIDIA_SMI: str = "nvidia-smi"
    NEURON_LS: str = "neuron-ls"
    # if the utility is not present the return from the whereis command is like
    # 'neuron-ls:\n' or 'nvidia-smi:\n' otherwise it is like 
    # 'nvidia-smi: /usr/bin/nvidia-smi /usr/share/man/man1/nvidia-smi.1.gz\n'
    utility_present = lambda x: subprocess.check_output(["whereis", x]).decode() != f"{x}:\n"
    is_nvidia = utility_present(NVIDIA_SMI)
    is_neuron = utility_present(NEURON_LS)
    logger.info(f"is_nvidia={is_nvidia}, is_neuron={is_neuron}")
    if is_nvidia is True:
        accelerator = constants.ACCELERATOR_TYPE.NVIDIA
    else:
        accelerator = constants.ACCELERATOR_TYPE.NEURON

    if is_nvidia is True:
        model_copies_as_int, devices_per_model = _get_model_copies_to_start_nvidia(tp_degree, model_copies)
    else:
        model_copies_as_int, devices_per_model = _get_model_copies_to_start_neuron(tp_degree, model_copies)
    
    docker_compose, nginx_config, per_container_info_list = _create_config_files(model_id,
                                                                                 model_copies_as_int,
                                                                                 devices_per_model,
                                                                                 image,
                                                                                 tp_degree,
                                                                                 user,
                                                                                 shm_size,
                                                                                 env_as_list,
                                                                                 accelerator)

    yaml.Dumper.ignore_aliases = lambda self, data: True
    docker_compose_yaml = yaml.dump(docker_compose)

    # create the docker compose and nginx.conf file in the top level
    # directory path for this model
    dir_path = os.path.join(dir_path, Path(model_id).name)
    os.makedirs(dir_path, exist_ok=True)
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

        # write config.properties
        conf_dir = os.path.join(pc["dir_path_on_host"], "conf")
        os.makedirs(conf_dir, exist_ok=True)
        cp_fpath = os.path.join(conf_dir, "config.properties")
        logger.info(f"writing {pc['config_properties']} to {cp_fpath}")
        Path(cp_fpath).write_text(pc['config_properties'])

    return model_copies_as_int
