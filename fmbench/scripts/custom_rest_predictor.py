import os
import json
import math
import time
import boto3
import logging
import requests
import pandas as pd
from datetime import datetime
from fmbench.scripts import constants
from fmbench.utils import count_tokens
from typing import Dict, Optional, List
from fmbench.scripts.fmbench_predictor import (FMBenchPredictor,
                                               FMBenchPredictionResponse)

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomRestPredictor(FMBenchPredictor):
    """
    This is a custom rest predictor that does a POST request on the endpoint
    specified in the configuration file with custom headers, authentication parameters
    and the model_id.
    """
    def __init__(self,
                 endpoint_name: str,
                 inference_spec: Optional[Dict],
                 metadata: Optional[Dict]):
        try:
            self._endpoint_name: str = endpoint_name
            self._inference_spec: Dict = inference_spec 
        except Exception as e:
            logger.error(f"create_predictor, exception occured while creating predictor "
                         f"for endpoint_name={self._endpoint_name}, exception={e}")
        logger.info(f"_endpoint_name={self._endpoint_name}, _inference_spec={self._inference_spec}")

    def get_prediction(self, payload: Dict) -> FMBenchPredictionResponse:
        # Initialize some variables, including the response, latency, streaming variables, prompt and completion tokens.
        response_json: Optional[Dict] = None
        response: Optional[str] = None
        latency: Optional[float] = None
        TTFT: Optional[float] = None
        TPOT: Optional[float] = None
        TTLT: Optional[float] = None
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        generated_text: Optional[str] = None
        
        try:
            # define the generation config, custom headers and model id from the inference spec
            # that will be used in the payload while FMBench makes predictions on the model endpoint
            inference_param_set = self._inference_spec.get("parameters")
            headers = self._inference_spec.get("headers")
            model_id = self._inference_spec.get("model_id")
            # Prepare the request body - in this request body, we are providing the generation config, prompt
            # and the model id as given in the inference spec within the FMBench config file
            if inference_param_set:
                request_body = {
                    "model_id": model_id,
                    "prompt": payload['inputs'],
                } | inference_param_set
            else:
                logger.info(f"Using the request body without the generation_config variable")
                request_body = {
                    "model_id": model_id,
                    "prompt": payload['inputs']
                }
                
            prompt_tokens = count_tokens(payload["inputs"])
            # Start the timer to measure the latency of the prediction made to the endpoint
            st = time.perf_counter()
            # Make POST request including the headers, the request body, and the endpoint url.
            response = requests.post(
                self._endpoint_name,
                headers=headers,
                json=request_body
            )
            # measure the total latency to make the POST request to the endpoint
            latency = time.perf_counter() - st
            response.raise_for_status()
            response_data = response.json()
            # Extract the generated text from the completions array
            if response_data.get("completions"):
                generated_text = response_data["completions"][0].get("text", "")
            response_json = dict(generated_text=generated_text)
            # Get completion tokens from the usage information if available
            # otherwise fall back to counting tokens
            if response_data.get("usage"):
                completion_tokens = response_data["usage"].get("completion_tokens")
                prompt_tokens = response_data["usage"].get("prompt_tokens")
            else:
                completion_tokens = count_tokens(generated_text) 
        except requests.exceptions.RequestException as e:
            logger.error(f"get_prediction, exception occurred while getting prediction for payload={payload} "
                        f"from predictor={self._endpoint_name}, response={response}, exception={e}")
        return FMBenchPredictionResponse(
            response_json=response_json,
            latency=latency,
            time_to_first_token=TTFT,
            time_per_output_token=TPOT,
            time_to_last_token=TTLT,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens
        )
        
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

    def shutdown(self) -> None:
        """Represents the function to shutdown the predictor
           cleanup the endpooint/container/other resources
        """
        return None
    
    @property
    def inference_parameters(self) -> Dict:
        """The inference parameters property."""
        return self._inference_spec.get("parameters")

    @property
    def platform_type(self) -> Dict:
        """The inference parameters property."""
        return constants.PLATFORM_EXTERNAL
    
def create_predictor(endpoint_name: str, inference_spec: Optional[Dict], metadata: Optional[Dict]):
    return CustomRestPredictor(endpoint_name, inference_spec, metadata)
