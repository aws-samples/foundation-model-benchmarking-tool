"""
ollama specific code
"""
import logging
from fmbench.scripts.inference_containers.utils import (STOP_AND_RM_CONTAINER,
                                                        FMBENCH_MODEL_CONTAINER_NAME)

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def create_script(region, image_uri, model_id, model_name, env_str, privileged_str, hf_token, directory):
    """
    Script for running the docker container for the inference server
    """
    script = f"""#!/bin/sh

        {STOP_AND_RM_CONTAINER}

        # Check for CUDA devices and enable them
        echo "Checking for CUDA devices..."
        if command -v nvidia-smi &> /dev/null; then
            gpu_count=`nvidia-smi --list-gpus | wc -l`
            if [ \$gpu_count -gt 0 ]; then
                gpu_count=\$((gpu_count - 1))
                gpus_to_enable=`seq -s, 0 \$gpu_count`
                export CUDA_VISIBLE_DEVICES=\$gpus_to_enable
                echo "CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
                echo "Found and enabled \$((gpu_count + 1)) GPU(s)"
            else
                echo "No GPUs found"
                export CUDA_VISIBLE_DEVICES=""
            fi
        else
            echo "nvidia-smi not found. Running in CPU mode"
            export CUDA_VISIBLE_DEVICES=""
        fi
    
        # Install ollama
        curl -fsSL https://ollama.com/install.sh | sh
        
        #Stop ollama
        systemctl stop ollama.service

        # Pull the specified model using Ollama
        echo "Pulling model: {model_id}..."
        ollama pull {model_id}
        if [ $? -ne 0 ]; then
            echo "Error: Failed to pull model {model_id}."
            exit 1
        fi

        # Serve the specified model using Ollama
        echo "Starting to serve model: {model_id}..."
        ollama serve {model_id} &
        if [ $? -ne 0 ]; then
            echo "Error: Failed to serve model {model_id}."
            exit 1
        fi

        echo "Successfully serving model_id = {model_id} with Ollama."
    """
    return script
