import os
import json
import copy
import time
import logging
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from fmbench.utils import count_tokens
from fmbench.scripts import constants
from fmbench.scripts.stream_responses import get_response_stream
from fmbench.scripts.sagemaker_metrics import get_endpoint_metrics
from fmbench.scripts.fmbench_predictor import (FMBenchPredictor,
                                               FMBenchPredictionResponse)
                                            
# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from enum import Enum

class CONTAINER_TYPE(str, Enum):
    DJL = 'djl'
    VLLM = 'vllm'

class EC2Predictor(FMBenchPredictor):
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
        container_type: Optional[str] = None
        # get the prompt for the EKS endpoint
        prompt: str = payload['inputs']
        # represents the number of tokens in the prompt payload
        prompt_tokens = count_tokens(payload['inputs'])
        try:
            st = time.perf_counter()
            split_input_and_inference_params: Optional[bool] = None
            if self._inference_spec is not None:
                split_input_and_inference_params = self._inference_spec.get("split_input_and_parameters")
                container_type = self._inference_spec.get("container_type", constants.CONTAINER_TYPE_DJL)
                logger.info(f"split input parameters is: {split_input_and_inference_params}, "
                            f"container_type={container_type}")
            # this is the POST request to the endpoint url for invocations that 
            # is given to you as you deploy a model on EC2 using the DJL serving stack
            if container_type == constants.CONTAINER_TYPE_DJL:
                payload = payload | dict(parameters=self._inference_spec["parameters"])
                response = requests.post(self._endpoint_name, json=payload)
            elif container_type == constants.CONTAINER_TYPE_VLLM:
                # vllm uses prompt rather than input and then
                # the code in the calling function still expects input
                # so make a copy
                payload2 = copy.deepcopy(payload)
                payload2['prompt'] = payload2.pop('inputs')
                payload2 = payload2 | self._inference_spec["parameters"]
                response = requests.post(self._endpoint_name, json=payload2)
            else:
                raise ValueError("container_type={container_type}, dont know how to handle this") 

            # record the latency for the response generated
            latency = time.perf_counter() - st
            
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
        namespace = "AWS/EC2"
        return get_endpoint_metrics("", namespace, "", start_time, end_time)
        

    @property
    def inference_parameters(self) -> Dict:
        """The inference parameters property."""
        return self._inference_spec.get("parameters")

def create_predictor(endpoint_name: str, inference_spec: Optional[Dict], metadata: Optional[Dict]):
    return EC2Predictor(endpoint_name, inference_spec, metadata)
