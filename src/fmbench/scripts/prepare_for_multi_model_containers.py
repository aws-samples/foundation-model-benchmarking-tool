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


def _handle_triton_serving_properties_and_inf_params(triton_dir: str, 
                                                     tp_degree: int, 
                                                     batch_size: int, 
                                                     inference_parameters: Dict, 
                                                     hf_model_id: str):
    """
    Takes the triton files: config.pbtxt, model.json, model.py and substitutes the batch size, tensor parallel degree
    from the configuration file into those files before the model repository is prepared within the container
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
                    # Replace the inference parameters with the inference parameters from
                    # the configuration file
                    if "on_device_embedding" in content.get("neuron_config", {}):
                        logger.info(f"Replacing on_device_embedding in {file_path} with inference_parameters.")
                        content["neuron_config"]["on_device_embedding"] = inference_parameters
                    else:
                        logger.warning(f"on_device_embedding not found in {file_path}.")
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
                else:
                    logger.info(f"No substitutions needed for {file_path}")
    except Exception as e:
        raise Exception(f"Error occurred while preparing files for the triton model container: {e}")

def _create_config_files(model_id: str,
                         num_model_copies: int,
                         devices_per_model: int,
                         tp_degree: int,
                         image: str,
                         user: str,
                         shm_size: str,
                         env: List,
                         accelerator: constants.ACCELERATOR_TYPE) -> Tuple:

    """
    Creates the docker compose yml file, nginx config file, these are common to all containers
    and then creates individual config.properties files
    """

    """
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

    # load balancer section
    # an lb is needed only if model copies > 1 because if it is 1 then
    # we can just have the model container list on 8080
    if num_model_copies > 1:
        lb = dict(image="nginx:alpine",
                 container_name="fmbench_model_container_load_balancer",
                 ports=["8080:80"],
                 volumes=["./nginx.conf:/etc/nginx/nginx.conf:ro"],
                 depends_on=[],
                 deploy=dict(placement=dict(constraints=['node.role == manager'])))
    else:
        logger.info(f"num_model_copies={num_model_copies}, not going to add a load balancer")
        lb = None

    if lb is not None:
        services = dict(loadbalancer=lb)
    else:
        services = dict()

    cnames = []
    nginx_server_lines = []
    per_container_info_list = []
    base_port = 8080 if lb is not None else 8079
    for i in range(num_model_copies):        
        cname = f"fmbench_model_container_{i+1}"
        cnames.append(cname)
        port = base_port + i + 1
        logger.info(f"container {i+1} will run on port {port}")
        if user == constants.CONTAINER_TYPE_TRITON:
            internal_http_port = 8000
            cname = f"fmbench_model_container_{i+1}"
            # if it is a triton container, then point the cname to the internal http port for 
            # triton 
            nginx_server_lines.append(f"        server {cname}:{internal_http_port};")
        elif ((user == constants.CONTAINER_TYPE_DJL) or (user == constants.CONTAINER_TYPE_VLLM)):
            nginx_server_lines.append(f"        server {cname}:{port};")
        else:
            logger.error(f"No load balancer option to be provided")
        
        device_offset = devices_per_model * i
        if accelerator == constants.ACCELERATOR_TYPE.NEURON:
            device_name = constants.ACCELERATOR_TYPE.NEURON.value
            devices = [f"/dev/{device_name}{j + device_offset}:/dev/{device_name}{j}" for j in range(devices_per_model)]
            # nothing extra, only nvidia has an extra env var for GPUs
            extra_env = []
        else:
            devices = None
            # add visible devices via env var
            gpus=",".join([str(j+device_offset) for j in range(devices_per_model)])
            extra_env = [f"NVIDIA_VISIBLE_DEVICES={gpus}"]

        if user == constants.CONTAINER_TYPE_TRITON:
            current_dir: str = os.path.dirname(os.path.realpath(__file__))
            triton_content: str = os.path.join(current_dir, "triton")
            logger.info(f"All triton model content is in: {triton_content}")
            dir_path_on_host: str = f"/home/ubuntu/{Path(model_id).name}"
            
            # Copy Triton content to each i{i+1} directory
            # The triton content directory contains: model.py, model.json, config.pbtxt
            # and a script that creates these files in the container model repository. 
            # The TP degree, batch size, and inference parameters are configured from the configuration
            # file into the triton container
            for i in range(num_model_copies):  
                instance_dir = os.path.join(dir_path_on_host, f"i{i+1}")
                triton_instance_dir = os.path.join(instance_dir, "triton")
                os.makedirs(triton_instance_dir, exist_ok=True)
                
                # Copy all files from triton_content to the instance's triton directory
                for item in os.listdir(triton_content):
                    s = os.path.join(triton_content, item)
                    d = os.path.join(triton_instance_dir, item)
                    if os.path.isfile(s):
                        shutil.copy2(s, d)
                    elif os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                # give permissions to run the script that creates the model repository in the triton model container
                os.chmod(os.path.join(triton_instance_dir, "triton-transformers-neuronx.sh"), 0o755)
            triton_files: List[str] = [f for f in os.listdir(triton_content) if os.path.isfile(os.path.join(triton_content, f))]
            logger.info(f"Files in triton content that have been copied to each instance directory: {triton_files}")
            volumes = [f"{dir_path_on_host}/i{i+1}/triton:/scripts/triton:rw",
                        f"{dir_path_on_host}/i{i+1}/triton:/triton:rw",
                        f"{dir_path_on_host}/snapshots:/snapshots:rw"]
            ports = [f"{port}:{internal_http_port}"]
            model_container_download_script: str = '/scripts/triton/triton-transformers-neuronx.sh'
        else:
            dir_path_on_host = f"/home/ubuntu/{Path(model_id).name}/i{i+1}"
            volumes = [f"{dir_path_on_host}:/opt/ml/model:ro",
                   f"{dir_path_on_host}/conf:/opt/djl/conf:ro",
                   f"{dir_path_on_host}/model_server_logs:/opt/djl/logs"]
            ports = [f"{port}:{port}"]

        service = {cname: { "image": image, 
                            "container_name": cname,
                            "user": user if (user == constants.CONTAINER_TYPE_DJL or user == constants.CONTAINER_TYPE_VLLM) else '', 
                            "shm_size": shm_size,
                            "devices": devices,
                            "environment": env + extra_env,
                            "volumes": volumes,
                            "ports": ports,
                            "deploy": {"restart_policy": {"condition": "on-failure"}} }}

        if accelerator == constants.ACCELERATOR_TYPE.NVIDIA:
            service[cname]['runtime'] = constants.ACCELERATOR_TYPE.NVIDIA.value
            _ = service[cname].pop("devices")
        elif user == constants.CONTAINER_TYPE_TRITON:
            logger.info(f"This is a triton image uri, using an entry point command which contains the model repository contents")
            service[cname]['command'] = f"{model_container_download_script} {model_id} {Path(model_id).name}"
        services = services | service

        # config.properties
        config_properties = f"""
inference_address=http://0.0.0.0:{port}
management_address=http://0.0.0.0:{port}
cluster_address=http://0.0.0.0:8888
model_store=/opt/ml/model
load_models=ALL
#model_url_pattern=.*
    """

        # save info specific to each container instance
        per_container_info_list.append(dict(dir_path_on_host=dir_path_on_host, 
                                            config_properties=config_properties))

    if lb is not None:
        lb["depends_on"] = cnames
    docker_compose = dict(services=services)
    
    # nginx.conf file
    nginx_server_lines = "\n".join(nginx_server_lines)
    nginx_config = """
### Nginx Load Balancer
events {}
http {
    upstream djlcluster {
__nginx_server_lines__
    }
    server {
        listen 80;
        location / {
            proxy_pass http://djlcluster;
        }
    }
}
"""
    if lb is not None:
        nginx_config = nginx_config.replace("__nginx_server_lines__", nginx_server_lines)
    else:
        nginx_config = None

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
    tp_degree: int = inference_params.get('tp_degree', None)
    batch_size: int = inference_params.get('batch_size', None)
    inference_parameters: Dict = inference_params.get('parameters', None)
    if user == constants.CONTAINER_TYPE_TRITON:
        current_dir: str = os.path.dirname(os.path.realpath(__file__))
        triton_content: str = os.path.join(current_dir, "triton")
        # handles custom tensor pd, batch size into the model repository files
        _handle_triton_serving_properties_and_inf_params(triton_content, tp_degree, batch_size, inference_parameters, model_id)

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
                                                                                 tp_degree,
                                                                                 image,
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