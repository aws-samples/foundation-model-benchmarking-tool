# script that prepares a directory structure to run multiple copies of a model
# using the DJL container. This code is written for Neuron for now but would be modified
# to run on GPUs as well and then maybe other inference containers as well.
# It creates a docker_compose.yml file that creates multiple containers and a load balancer.
# which provides a single endpoint to external applications that want to use these containers.

import os
import json
import yaml
import docker
import logging
import subprocess
from pathlib import Path
import fmbench.scripts.constants as constants
from typing import Dict, List, Optional, Tuple, Union

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def _create_config_files(model_id: str,
                         num_model_copies: int,
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
        - HF_TOKEN=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
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
        - HF_TOKEN=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
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

    # per model service info to be put in docker compose
    if accelerator == constants.ACCELERATOR_TYPE.NEURON:
        num_devices_per_model = int(tp_degree / 2)
    else:
        num_devices_per_model = tp_degree

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
        if lb is not None:
            nginx_server_lines.append(f"        server {cname}:{port};")
        
        device_offset = num_devices_per_model * i
        if accelerator == constants.ACCELERATOR_TYPE.NEURON:
            device_name = constants.ACCELERATOR_TYPE.NEURON.value
            devices = [f"/dev/{device_name}{j + device_offset}:/dev/{device_name}{j}" for j in range(num_devices_per_model)]
            # nothing extra, only nvidia has an extra env var for GPUs
            extra_env = []
        else:
            # device_name = constants.ACCELERATOR_TYPE.NVIDIA.value
            # capabilities = '[gpu]'
            # device_ids = [str(j + device_offset) for j in range(num_devices_per_model)]
            # devices = [dict(driver=device_name, device_ids=device_ids, capabilities=capabilities)]
            devices = None
            # add visible devices via env var
            gpus=",".join([str(j+device_offset) for j in range(num_devices_per_model)])
            extra_env = [f"NVIDIA_VISIBLE_DEVICES={gpus}"]
            
        
        dir_path_on_host = f"/home/ubuntu/{model_id}/i{i+1}"
        volumes = [f"{dir_path_on_host}:/opt/ml/model:ro",
                   f"{dir_path_on_host}/conf:/opt/djl/conf:ro",
                   f"{dir_path_on_host}/model_server_logs:/opt/djl/logs"]
        
        service = {cname: { "image": image, 
                            "container_name": cname,
                            "user": user, 
                            "shm_size": shm_size,
                            "devices": devices,
                            "environment": env + extra_env,
                            "volumes": volumes,
                            "ports": [f"{port}:{port}"],
                            "deploy": {"restart_policy": {"condition": "on-failure"}} }}
        if accelerator == constants.ACCELERATOR_TYPE.NVIDIA:
            service[cname]['runtime'] = constants.ACCELERATOR_TYPE.NVIDIA.value
            _ = service[cname].pop("devices")
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

def _get_model_copies_to_start_neuron(tp_degree: int, num_model_copies: Union[str, int]) -> Optional[int]:
    logger.info(f"_get_model_copies_to_start_neuron, tp_degree={tp_degree}, num_model_copies={num_model_copies}")
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

    # tensor parallelism requires as many neuron cores as tp degree
    # and the number of devices required is half of number of cores
    neuron_devices_needed_per_model_copy = int(tp_degree / 2)
    model_copies_possible = int(num_neuron_devices/neuron_devices_needed_per_model_copy)
    # if num_model_copies is max then load as many copies as possible
    if isinstance(num_model_copies, str):
        if num_model_copies == "max":
            num_model_copies = model_copies_possible
            logger.info(f"num_model_copies was set to \"max\", "
                        f"this instance can support a max of {num_model_copies} model copies, "
                        f"going to load {num_model_copies} copies")
        else:            
            logger.info(f"num_model_copies set to a str value={num_model_copies}, "
                        f"will see if we can load these many models")
            num_model_copies = int(num_model_copies)
    else:
        logger.info(f"num_model_copies set to a numerical value={num_model_copies}, "
                    f"will see if we can load these many models")

    num_devices_needed = num_model_copies * neuron_devices_needed_per_model_copy
    
    logger.info(f"num_model_copies={num_model_copies}, tp_degree={tp_degree},\n"
                f"num_neuron_devices={num_neuron_devices}, num_neuron_cores={num_neuron_cores},\n"
                f"neuron_devices_needed_per_model_copy={neuron_devices_needed_per_model_copy}, num_devices_needed={num_devices_needed},\n"
                f"model_copies_possible={model_copies_possible}")

    if model_copies_possible < num_model_copies:
        logger.error(f"num_model_copies={num_model_copies} but model_copies_possible={model_copies_possible}, "
                     f"setting num_model_copies to max possible for this instance which is {model_copies_possible}")
        num_model_copies = model_copies_possible
    else:
        logger.error(f"num_model_copies={num_model_copies} and model_copies_possible={model_copies_possible}, "
                     f"it is possible to run {num_model_copies} models, going with that")

    return num_model_copies

def _get_model_copies_to_start_nvidia(tp_degree: int, num_model_copies: Union[str, int]) -> Optional[int]:
    logger.info(f"_get_model_copies_to_start_nvidia, tp_degree={tp_degree}, num_model_copies={num_model_copies}")
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
    
    # tensor parallelism requires as many gpus as tp degree
    gpus_needed_per_model_copy = tp_degree
    model_copies_possible = int(num_gpus/gpus_needed_per_model_copy)
    # if num_model_copies is max then load as many copies as possible
    if isinstance(num_model_copies, str):
        if num_model_copies == "max":
            num_model_copies = model_copies_possible
            logger.info(f"num_model_copies was set to \"max\", "
                        f"this instance can support a max of {num_model_copies} model copies, "
                        f"going to load {num_model_copies} copies")
        else:            
            logger.info(f"num_model_copies set to a str value={num_model_copies}, "
                        f"will see if we can load these many models")
            num_model_copies = int(num_model_copies)
    else:
        logger.info(f"num_model_copies set to a numerical value={num_model_copies}, "
                    f"will see if we can load these many models")

    num_gpus_needed = num_model_copies * gpus_needed_per_model_copy
    
    logger.info(f"num_model_copies={num_model_copies}, tp_degree={tp_degree},\n"
                f"num_gpus={num_gpus},\n"
                f"gpus_needed_per_model_copy={gpus_needed_per_model_copy}, num_gpus_needed={num_gpus_needed},\n"
                f"model_copies_possible={model_copies_possible}")

    if model_copies_possible < num_model_copies:
        logger.error(f"num_model_copies={num_model_copies} but model_copies_possible={model_copies_possible}, "
                     f"setting num_model_copies to max possible for this instance which is {model_copies_possible}")
        num_model_copies = model_copies_possible
    else:
        logger.error(f"num_model_copies={num_model_copies} and model_copies_possible={model_copies_possible}, "
                     f"it is possible to run {num_model_copies} models, going with that")

    return num_model_copies


def prepare_docker_compose_yml(model_id: str,
                               num_model_copies: Union[int, str],
                               tp_degree: int,
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
        num_model_copies = _get_model_copies_to_start_nvidia(tp_degree, num_model_copies)
    else:
        num_model_copies = _get_model_copies_to_start_neuron(tp_degree, num_model_copies)
    
    docker_compose, nginx_config, per_container_info_list = _create_config_files(model_id,
                                                                                 num_model_copies,
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
    dir_path = os.path.join(dir_path, model_id)
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
        logger.info(f"num_model_copies={num_model_copies}, nginx_config={nginx_config}, "
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

    return num_model_copies
