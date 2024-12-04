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
from fmbench.scripts.ec2_metrics import collect_ec2_metrics, stop_collect
from fmbench.scripts.inference_containers.utils import get_accelerator_type
from fmbench.scripts.fmbench_predictor import (FMBenchPredictor,
                                               FMBenchPredictionResponse)
from fmbench.scripts.inference_containers.utils import (STOP_AND_RM_CONTAINER,
                                                        FMBENCH_MODEL_CONTAINER_NAME)
                                            
# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EC2Predictor(FMBenchPredictor):
    # overriding abstract method
    def __init__(self,
                 endpoint_name: str,
                 inference_spec: Optional[Dict], 
                 metadata: Optional[Dict]):
        try:
            self._endpoint_name: str = endpoint_name
            self._inference_spec: Dict = inference_spec
            self._accelerator = get_accelerator_type()
            # Start collecting EC2 metrics. This will be called once.
            collect_ec2_metrics()
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
                st = time.perf_counter()
                response = requests.post(self._endpoint_name, json=payload)
                # record the latency for the response generated
                latency = time.perf_counter() - st                
                #logger.info(f"full_output={response.text}")
                response.raise_for_status()
                """
                the output is of the form
                {"generated_text": "\n\nSuining had a population of 658,798 in 2002."}
                we only need the generated_text field from this
                """
                full_output = json.loads(response.text).get("generated_text")
                if full_output is None:
                    logger.error(f"failed to extract output from response text, response text = \"{response.text}\"")      
            elif container_type == constants.CONTAINER_TYPE_TRITON:
                if self._accelerator == constants.ACCELERATOR_TYPE.NEURON:
                    triton_payload = dict(text_input=payload["inputs"],
                                          sampling_parameters=json.dumps(self._inference_spec["parameters"]))
                else:
                    triton_payload = dict(text_input=payload["inputs"]) | self._inference_spec["parameters"]

                logger.info(f"Endpoint name is: {self._endpoint_name}, triton payload is: {triton_payload}")
                st = time.perf_counter()
                response = requests.post(self._endpoint_name, json=triton_payload)
                # record the latency for the response generated
                latency = time.perf_counter() - st
                response.raise_for_status()
                response_json = json.loads(response.text)
                full_output = response_json['text_output']
            elif container_type == constants.CONTAINER_TYPE_VLLM:
                # vllm uses prompt rather than input and then
                # the code in the calling function still expects input
                # so make a copy
                payload2 = copy.deepcopy(payload)
                payload2['prompt'] = payload2.pop('inputs')
                payload2 = payload2 | self._inference_spec["parameters"]
                st = time.perf_counter()
                response = requests.post(self._endpoint_name, json=payload2)
                # record the latency for the response generated
                latency = time.perf_counter() - st
                response.raise_for_status()
                """
                the output is of the form
                {
                    "id": "cmpl-e9d590543c374d828d724a228fae2604",
                    "object": "text_completion",
                    "created": 1731734600,
                    "model": "meta-llama/Llama-3.2-1b-Instruct",
                    "choices": [
                        {
                            "index": 0,
                            "text": "\n\nThey are both tennis players.",
                            "logprobs": null,
                            "finish_reason": "stop",
                            "stop_reason": null,
                            "prompt_logprobs": null
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1392,
                        "total_tokens": 1400,
                        "completion_tokens": 8,
                        "prompt_tokens_details": null
                    }
                }
                we only need the response field from this
                """
                full_output = None
                choices = json.loads(response.text).get("choices")
                if choices is not None:
                    full_output = choices[0].get("text")
                if full_output is None:
                    logger.error(f"failed to extract output from response text, response text = \"{response.text}\"")
            elif container_type == constants.CONTAINER_TYPE_OLLAMA:
                # ollama uses prompt rather than input and then
                # the code in the calling function still expects input
                # so make a copy
                payload2 = copy.deepcopy(payload)
                payload2['prompt'] = payload2.pop('inputs')
                payload2 = payload2 | self._inference_spec["parameters"]
                st = time.perf_counter()
                response = requests.post(self._endpoint_name, json=payload2)
                # record the latency for the response generated
                latency = time.perf_counter() - st
                response.raise_for_status()
                """
                the output is of the form
                {"model":"llama3.1:8b",
                 "created_at":"2024-10-27T13:44:12.400070826Z",
                 "response":"Once upon a time, in a small village nestled",
                 "done":true,
                 "done_reason":"length",
                 "context":[128006,882,128007,271,73457,757,264,3446,128009,128006,78191,128007,271,12805,5304,264,892,11,304,264,2678,14458,89777],
                 "total_duration":195361027,
                 "load_duration":25814720,
                 "prompt_eval_count":14,
                 "prompt_eval_duration":14601000,
                 "eval_count":10,
                 "eval_duration":96146000}
                we only need the response field from this
                """
                full_output = json.loads(response.text).get("response")
                if full_output is None:
                    logger.error(f"failed to extract output from response text, response text = \"{response.text}\"")              
            else:
                raise ValueError("container_type={container_type}, dont know how to handle this") 

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

    def shutdown(self) -> None:
        """Represents the function to shutdown the predictor
           cleanup the endpooint/container/other resources
        """
        # Stop collecting EC2 metrics either when the model container is stopped and removed, 
        # or once the benchmarking test has completed 
        stop_collect()

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
        """
        Retrieves EC2 system metrics from the CSV file generated during metric collection.

        Args:
            start_time (datetime): Start time of metrics collection
            end_time (datetime): End time of metrics collection
            period (int): Sampling period in seconds (default 60)
        
        Returns:
            pd.DataFrame: DataFrame containing system metrics
        """
        try:
            filtered_metrics: Optional[pd.DataFrame] = None
            # Read the CSV file generated by collect_ec2_metrics()
            metrics_df = pd.read_csv(constants.EC2_SYSTEM_METRICS_FNAME, parse_dates=['timestamp'])
            filtered_metrics = metrics_df[(metrics_df['timestamp'] >= start_time) & 
                                            (metrics_df['timestamp'] <= end_time)]
        except FileNotFoundError:
            logger.warning("Metrics CSV file containin the EC2 metrics not found.")
            filtered_metrics = None
        except Exception as e:
            logger.error(f"Error retrieving metrics: {e}")
        return filtered_metrics

    @property
    def inference_parameters(self) -> Dict:
        """The inference parameters property."""
        return self._inference_spec.get("parameters")
    
    @property
    def platform_type(self) -> Dict:
        """The inference parameters property."""
        return constants.PLATFORM_EC2


def create_predictor(endpoint_name: str, inference_spec: Optional[Dict], metadata: Optional[Dict]):
    return EC2Predictor(endpoint_name, inference_spec, metadata)
