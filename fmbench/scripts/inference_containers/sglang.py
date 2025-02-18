#!/usr/bin/env python3
"""
sglang.py

This module provides a helper function to create a shell script for deploying
the SGLang server with the DeepSeek-R1 models on sglang.

The generated script will:
  - Stop and remove any existing container with the same name.
  - Pull the SGLang Docker image.
  - Launch the container with the hardcoded model and runtime parameters.
"""

import logging
from fmbench.scripts.inference_containers.utils import (STOP_AND_RM_CONTAINER,
                                                        FMBENCH_MODEL_CONTAINER_NAME)

logging.basicConfig(
    format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def create_script(region, image_uri, model_id, model_name, env_str, privileged_str, hf_token, directory, cli_params=""):
    """
    Create a shell script for deploying the SGLang server.

    Parameters:
        region (str): Deployment region (not used directly here).
        image_uri (str): Docker image URI for SGLang (e.g. "lmsysorg/sglang:latest").
        model_id (str): The Hugging Face model identifier 
                        (e.g. "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").
        model_name (str): A name for the model (used in naming the container).
        env_str (str): Additional environment variables (not used here).
        privileged_str (str): Docker flag for privileged mode (e.g. "--privileged").
        hf_token (str): Hugging Face token (not required for SGLang).
        directory (str): Directory where the deployment script will reside.
        cli_params (str): Extra CLI parameters to pass to the SGLang server command.

    Returns:
        str: The content of the shell deployment script.
    """
    server_port = "30000"
    # The model_id and image_uri are passed in, but here we assume they are hardcoded values.
    script = f"""#!/bin/sh
# Stop and remove any existing container with the same name
{STOP_AND_RM_CONTAINER}

echo "Pulling SGLang Docker image: {image_uri}"
docker pull {image_uri}

echo "Launching SGLang container with model: {model_id} on port: {server_port}"
docker run -d --rm --gpus all --shm-size 32g \\
    -p {server_port}:{server_port} \\
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \\
    --ipc host --network host {privileged_str} \\
    --name {FMBENCH_MODEL_CONTAINER_NAME} {image_uri} \\
    python3 -m sglang.launch_server \\
    --model {model_id} --trust-remote-code --port {server_port} {cli_params}

"""
    logger.info("SGLang deployment script created.")
    return script
