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
    and the model_id. This rest predictor can be used with custom parameters. View an 
    example of the parameters passed in this config file: configs/byoe/config-byo-custom-rest-predictor.yml
    """
    def __init__(self,
                 endpoint_name: str,
                 inference_spec: Optional[Dict],
                 metadata: Optional[Dict]):
        try:
            """
            Initialize the endpoint name and the inference spec. The endpoint name here points to the
            endpoint name in the config file, which is the endpoint url to do a request.post on. The 
            inference spec contains the different auth, headers, inference parameters that are 
            passed into this script from the config file.
            """
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
        # Streaming can be enabled if the model is deployed on SageMaker or Bedrock
        TTFT: Optional[float] = None
        TPOT: Optional[float] = None
        TTLT: Optional[float] = None
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        # This is the generated text from the model prediction
        generated_text: Optional[str] = None
        
        try:
            # define the generation config, custom headers and model id from the inference spec
            # that will be used in the payload while FMBench makes predictions on the model endpoint
            inference_param_set = self._inference_spec.get("parameters")
            headers = self._inference_spec.get("headers")
            model_id = self._inference_spec.get("model_id")
            # Prepare the request body - in this request body, we are providing the generation config, prompt
            # and the model id as given in the inference spec within the FMBench config file. If the inference
            # parameter set does not exist, then just send in the request without the inference specifications
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
                # This is custom to the endpoint based on the completion format. This will change
                # based on how your inference container responds to requests.
                generated_text = response_data["completions"][0].get("text", "")
            response_json = dict(generated_text=generated_text)
            # Get completion tokens from the usage information if available
            # otherwise fall back to counting tokens
            if response_data.get("usage"):
                # This is assuming the response data contains a usage field with prompt and input tokens
                completion_tokens = response_data["usage"].get("completion_tokens")
                prompt_tokens = response_data["usage"].get("prompt_tokens")
                logger.info(f"Found 'usage' field in the response data. Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
            else:
                # This uses the count tokens function. If you have an hf tokenizer to be used, then 
                # place your hf_token.txt file in the fmbench-read/scripts directory and mention the 
                # hf model id in the experiments section of the config file in the "hf_tokenizer_model_id"
                # paramter. The count_tokens function will use that custom tokenizer. If you have a custom
                # tokenizer, then place the "tokenizer.json" and "config.json" files in the fmbench-read/scripts
                # directory and FMBench will use that. If none of these options are available, FMBench will use
                # the default 750-1000 tokens tokenizer.
                prompt_tokens = count_tokens(payload["inputs"])
                completion_tokens = count_tokens(generated_text) 
                logger.info(f"Using the default tokenizer to count the prompt and input tokens. Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
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
