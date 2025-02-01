"""
vllm specific code
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

        # Run the new Docker container with specified settings
        docker run -d {privileged_str} --rm --name={FMBENCH_MODEL_CONTAINER_NAME} --env "HF_TOKEN={hf_token}" --ipc=host -p 8000:8000 {env_str} {image_uri} --model {model_id}

        echo "started docker run in daemon mode"
    """
    return script
