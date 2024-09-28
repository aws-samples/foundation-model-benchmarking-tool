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
from fmbench.scripts.ec2_metrics import collect_ec2_metrics, stop_collect
from fmbench.scripts.neuron_metrics import start_collection, stop_collection, get_collected_data, reset_collection
from fmbench.scripts import constants
from fmbench.scripts.fmbench_predictor import (FMBenchPredictor,
                                               FMBenchPredictionResponse)
from fmbench.scripts.constants import ACCELERATOR_TYPE, IS_NEURON_INSTANCE
                                           
                                            
# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from enum import Enum

class CONTAINER_TYPE(str, Enum):
    DJL = 'djl'
    VLLM = 'vllm'

class EC2Predictor(FMBenchPredictor):
    _metrics_collection_started = False
    # overriding abstract method
    def __init__(self,
                 endpoint_name: str,
                 inference_spec: Optional[Dict], 
                 metadata: Optional[Dict]):
        try:
            self._endpoint_name: str = endpoint_name
            self._inference_spec: Dict = inference_spec
            self._accelerator_type: ACCELERATOR_TYPE = (ACCELERATOR_TYPE.NEURON 
                                            if IS_NEURON_INSTANCE(metadata.get('instance_type', ''))
                                            else ACCELERATOR_TYPE.NVIDIA)
        except Exception as e:
            logger.error(f"create_predictor, exception occured while creating predictor "
                         f"for endpoint_name={self._endpoint_name}, exception={e}")
            self._accelerator_type = ACCELERATOR_TYPE.NVIDIA
        logger.info(f"_endpoint_name={self._endpoint_name}, _inference_spec={self._inference_spec}, _accelerator_type={self._accelerator_type}")

    def get_prediction(self, payload: Dict) -> FMBenchPredictionResponse:
        if not self._metrics_collection_started:
            logger.info(f"Starting metrics collection for {self._accelerator_type}")
            if self._accelerator_type == ACCELERATOR_TYPE.NVIDIA:
                collect_ec2_metrics()
            else:
                start_collection()
            self._metrics_collection_started = True
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
            # Global flag us
            # collect_ec2_metrics()
            st = time.perf_counter()
            split_input_and_inference_params: Optional[bool] = None
            if self._inference_spec is not None:
                split_input_and_inference_params = self._inference_spec.get("split_input_and_parameters")
                container_type = self._inference_spec.get("container_type", constants.CONTAINER_TYPE_DJL)
                logger.debug(f"split input parameters is: {split_input_and_inference_params}, "
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

        logger.info("Entering get_metrics function")
        metrics_df = pd.DataFrame()
        
        try:
            if self._accelerator_type == ACCELERATOR_TYPE.NVIDIA:
                logger.info("Stopping EC2 metrics collection for NVIDIA")
                metrics_df = stop_collect()
            else:
                logger.info("Stopping Neuron metrics collection")
                stop_collection()
                metrics_df = get_collected_data()
                reset_collection()

            if not metrics_df.empty:
                metrics_df["EndpointName"] = self.endpoint_name
                metrics_df["ModelLatency"] = None
                logger.info(f"Metrics dataframe shape: {metrics_df.shape}")
            else:
                logger.warning("No metrics collected")

        except Exception as e:
            logger.error(f"Error occurred while collecting metrics: {e}", exc_info=True)

        return metrics_df

    @property
    def inference_parameters(self) -> Dict:
        """The inference parameters property."""
        return self._inference_spec.get("parameters")

def create_predictor(endpoint_name: str, inference_spec: Optional[Dict], metadata: Optional[Dict]):
    return EC2Predictor(endpoint_name, inference_spec, metadata)

