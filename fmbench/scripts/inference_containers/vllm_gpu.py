"""
vllm specific code
"""
import logging
from fmbench.scripts.inference_containers.utils import (STOP_AND_RM_CONTAINER,
                                                        FMBENCH_MODEL_CONTAINER_NAME)

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def create_script(region, image_uri, model_id, model_name, env_str, privileged_str, hf_token, directory, cli_params):
    """
    Script for running the docker container for the inference server
    """
    script = f"""#!/bin/sh

        {STOP_AND_RM_CONTAINER}
        # kill any existing vllm instance to free up gpu resources
        process_name="vllm"
        pid=$(pgrep -x "$process_name")
        if [ -n "$pid" ]; then
          echo "Killing process $process_name with PID $pid"
          kill -9 "$pid"
        else
          echo "No process named $process_name is running"
        fi
        sleep 10
        vllm serve {model_id} {cli_params} &

        echo "started docker run in daemon mode"
    """
    return script