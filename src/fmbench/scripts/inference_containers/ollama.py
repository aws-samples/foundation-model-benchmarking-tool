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
