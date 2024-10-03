import os
import json
import math
import time
import boto3
import logging
import requests
import datetime
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


class EKSPredictor(FMBenchPredictor):
    # overriding abstract method
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
        response_json: Optional[Dict] = None
        response: Optional[str] = None
        latency: Optional[float] = None
        TTFT: Optional[float] = None
        TPOT: Optional[float] = None
        TTLT: Optional[float] = None
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        timeout: Optional[int] = None
        auth: Optional[Dict] = None
        # get the prompt for the EKS endpoint
        prompt: str = payload['inputs']
        # represents the number of tokens in the prompt payload
        prompt_tokens = count_tokens(payload["inputs"])
        try:
            st = time.perf_counter()
            split_input_and_inference_params: Optional[bool] = None
            if self._inference_spec is not None:
                split_input_and_inference_params = self._inference_spec.get("split_input_and_parameters")
                logger.info(f"split input parameters is: {split_input_and_inference_params}")
                timeout = self._inference_spec.get("timeout", 180)
                auth = self._inference_spec.get("auth", None)
                logger.info(f"Initializing the timeout to: {timeout}, auth to configured authentication information")
                # Use the parameters that the model needs at inference. 
                # parameters: Optional[Dict] = inference_spec.get('parameters')

            # This endpoint only supports the GET method now, you can add support for POST method if your endpoint supports it.
            # As an example, the following URL is used with a query added at the end of the URL.
            # http://<NLB_DNS_NAME>/serve/infer?sentence=<prompt_payload>
            response = requests.get(f"{self._endpoint_name}/{prompt}",timeout=timeout) 

            latency = time.perf_counter() - st
            logger.info(response)
            # the response from the model on ray serve from the url prompt is given in this format. 
            # For other response types, change the logic below and add the response in the `generated_text` key within the response_json dict
            response.raise_for_status()
            full_output = response.text
            answer_only = full_output.replace(prompt, "", 1).strip('["]?\n')
            response_json = dict(generated_text=answer_only)
            # counts the completion tokens for the model using the default/user provided tokenizer
            completion_tokens = count_tokens(response_json.get("generated_text"))
        except requests.exceptions.RequestException as e:
            logger.error(f"get_prediction, exception occurred while getting prediction for payload={payload} "
                         f"from predictor={self._endpoint_name}, response={response}, exception={e}")
        return FMBenchPredictionResponse(response_json=response_json,
                                         latency=latency,
                                         time_to_first_token=TTFT,
                                         time_per_output_token=TPOT,
                                         time_to_last_token=TTLT,
                                         completion_tokens=completion_tokens,
                                         prompt_tokens=prompt_tokens)    

    @property
    def endpoint_name(self) -> str:
        """The endpoint name property."""
        return self._endpoint_name

    def get_metrics(self,
                    start_time: datetime,
                    end_time: datetime,
                    period: int = 60) -> pd.DataFrame:
        # not implemented
        return None

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
        return constants.PLATFORM_EC2
    
def create_predictor(endpoint_name: str, inference_spec: Optional[Dict], metadata: Optional[Dict]):
    return EKSPredictor(endpoint_name, inference_spec, metadata)
