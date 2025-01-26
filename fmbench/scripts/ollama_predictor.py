import os
import json
import copy
import time
import stat
import logging
import requests
import tempfile
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from fmbench.scripts import constants
from fmbench.utils import count_tokens
from fmbench.scripts.inference_containers.utils import get_accelerator_type
from fmbench.scripts.fmbench_predictor import (FMBenchPredictor,
                                               FMBenchPredictionResponse)
from fmbench.scripts.inference_containers.utils import (STOP_AND_RM_CONTAINER,
                                                        FMBENCH_MODEL_CONTAINER_NAME)
                                            
# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ollamaPredictor(FMBenchPredictor):
    # overriding abstract method
    def __init__(self,
                 endpoint_name: str,
                 inference_spec: Optional[Dict], 
                 metadata: Optional[Dict]):
        try:
            self._endpoint_name: str = endpoint_name
            self._inference_spec: Dict = inference_spec
            self._accelerator = get_accelerator_type()  # Not needed for Ollama, but keeping in case it's useful
        except Exception as e:
            logger.error(f"create_predictor, exception occurred while creating predictor "
                         f"for endpoint_name={self._endpoint_name}, exception={e}")
        logger.info(f"_endpoint_name={self._endpoint_name}, _inference_spec={self._inference_spec}")

    def get_prediction(self, payload: Dict) -> FMBenchPredictionResponse:
        response_json: Optional[Dict] = None
        response: Optional[str] = None
        latency: Optional[float] = None
        TTFT: Optional[float] = None
        TPOT: Optional[float] = None
        TTLT: Optional[float] = None
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None

        # Get the prompt for the Ollama endpoint
        prompt: str = payload['inputs']
        # Calculate the number of tokens in the prompt
        prompt_tokens = count_tokens(payload['inputs'])

        try:
            # Adjust the payload for Ollama
            payload2 = copy.deepcopy(payload)
            payload2['prompt'] = payload2.pop('inputs')  # Replace 'inputs' with 'prompt'
            payload2 = payload2 | self._inference_spec.get("parameters", {})  # Add inference parameters if present

            # Send POST request to the Ollama endpoint
            st = time.perf_counter()  # Start latency timer
            response = requests.post(self._endpoint_name, json=payload2)
            latency = time.perf_counter() - st  # Record the latency

            # Raise exception if the response failed
            response.raise_for_status()
            full_output = response.text

            # Extract the generated response (removing the original prompt if necessary)
            answer_only = full_output.replace(prompt, "", 1).strip('["]?\n')
            response_json = dict(generated_text=answer_only)

            # Count the completion tokens for the generated text
            completion_tokens = count_tokens(response_json.get("generated_text"))

        except Exception as e:
            logger.error(f"get_prediction, exception occurred while getting prediction for payload={payload} "
                         f"from predictor={self._endpoint_name}, response={response}, exception={e}")

        return FMBenchPredictionResponse(response_json=response_json,
                                         latency=latency,
                                         time_to_first_token=TTFT,
                                         time_per_output_token=TPOT,
                                         time_to_last_token=TTLT,
                                         completion_tokens=completion_tokens,
                                         prompt_tokens=prompt_tokens)
    
    
    def shutdown(self) -> None:
        """Represents the function to shutdown the predictor
           cleanup the endpooint/container/other resources
        """
        script = f"""#!/bin/sh

        {STOP_AND_RM_CONTAINER}
        """
        tmpdir = tempfile.gettempdir()
        script_file_path = os.path.join(tmpdir, "shutdown_container.sh")
        Path(script_file_path).write_text(script)
        st = os.stat(script_file_path)
        os.chmod(script_file_path, st.st_mode | stat.S_IEXEC)

        logger.info(f"going to run script {script_file_path}")
        subprocess.run(["bash", script_file_path], check=True)
        logger.info(f"done running bash script")
        return None
    
    @property
    def endpoint_name(self) -> str:
        """The endpoint name property."""
        return self._endpoint_name

    # The rest ep is deployed on an instance that incurs hourly cost hence, the calculcate cost function
    # computes the cost of the experiment on an hourly basis. If your instance has a different pricing structure
    # modify this function.
    def calculate_cost(self,
                    instance_type: str,
                    instance_count: int,
                    pricing: Dict,
                    duration: float,
                    prompt_tokens: int,
                    completion_tokens: int) -> float:
        """Calculate the cost of each experiment run."""
        experiment_cost: Optional[float] = None
        try:
            instance_based_pricing = pricing['pricing']['instance_based']
            hourly_rate = instance_based_pricing.get(instance_type, None)
            logger.info(f"the hourly rate for running on {instance_type} is {hourly_rate}, instance_count={instance_count}")
            # calculating the experiment cost for instance based pricing
            instance_count = instance_count if instance_count else 1
            experiment_cost = (hourly_rate / 3600) * duration * instance_count
        except Exception as e:
            logger.error(f"exception occurred during experiment cost calculation, exception={e}")
        return experiment_cost
    
    def get_metrics(self,
                    start_time: datetime,
                    end_time: datetime,
                    period: int = 60) -> pd.DataFrame:
        # not implemented
        return None

    @property
    def inference_parameters(self) -> Dict:
        """The inference parameters property."""
        return self._inference_spec.get("parameters")
    
    @property
    def platform_type(self) -> Dict:
        """The inference parameters property."""
        return constants.PLATFORM_EC2


def create_predictor(endpoint_name: str, inference_spec: Optional[Dict], metadata: Optional[Dict]):
    return ollamaPredictor(endpoint_name, inference_spec, metadata)
