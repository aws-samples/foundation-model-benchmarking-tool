"""
Utility functions common across inference containers
"""
import json
import logging
import subprocess
import fmbench.scripts.constants as constants
from typing import Dict, List, Optional, Tuple, Union

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

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


def get_accelerator_type() -> constants.ACCELERATOR_TYPE:
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
        return constants.ACCELERATOR_TYPE.NVIDIA
    else:
        return constants.ACCELERATOR_TYPE.NEURON


def get_model_copies_to_start_neuron(tp_degree: int, model_copies: str) -> Tuple[int, int]:
    logger.info(f"get_model_copies_to_start_neuron, tp_degree={tp_degree}, model_copies={model_copies}")
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

def get_model_copies_to_start_nvidia(tp_degree: int, model_copies: str) -> Tuple[int, int]:
    logger.info(f"get_model_copies_to_start_nvidia, tp_degree={tp_degree}, model_copies={model_copies}")
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
        logger.info(f"model_copies={model_copies} and model_copies_possible={model_copies_possible}, "
                     f"it is possible to run {model_copies} models, going with that")

    return model_copies, gpus_needed_per_model_copy

