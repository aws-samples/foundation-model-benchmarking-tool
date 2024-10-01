import os
import math
import json
import time
import boto3
import litellm
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from litellm import completion, token_counter
from fmbench.scripts.stream_responses import get_response_stream
from fmbench.scripts.fmbench_predictor import (FMBenchPredictor,
                                               FMBenchPredictionResponse)


# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExternalPredictor(FMBenchPredictor):
    # overriding abstract method
    def __init__(self,
                 endpoint_name: str,
                 inference_spec: Optional[Dict],
                 metadata: Optional[Dict]):
        try:
            # initialize private member variables
            self._endpoint_name = endpoint_name
            self._pt_model_id = None
            self._inference_spec = inference_spec
            self._aws_region = boto3.Session().region_name
            # litellm supports the following inference params as per
            # https://litellm.vercel.app/docs/completion/input
            self._temperature = 0.1
            self._max_tokens = 100
            self._top_p = 0.9
            # not used for now but kept as placeholders for future
            self._stream = None
            self._start = None
            self._stop = None
            # no caching of responses since we want every inference
            # call to be independant
            self._caching = False

            # override these defaults if there is an inference spec provided
            if inference_spec:
                parameters: Optional[Dict] = inference_spec.get('parameters')
                self._api_key: Optional[str] = inference_spec.get('api_key', None)
                logger.info(f"Open AI API key: {self._api_key}")
                if self._api_key is None:
                    logger.error(f"API key is not provided for external model id, please provide the key in the config file.")
                    return
                if parameters:
                    self._temperature = parameters.get('temperature', self._temperature)
                    self._max_tokens = parameters.get('max_tokens', self._max_tokens)
                    self._top_p = parameters.get('top_p', self._top_p)
                    self._stream = inference_spec.get("stream", self._stream)
                    self._stop = inference_spec.get("stop_token", self._stop)
                    self._start = inference_spec.get("start_token", self._start)
            self._response_json = {}
        except Exception as e:
            exception_msg = f"""exception while creating predictor/initializing variables
                            for endpoint_name={self._endpoint_name}, exception=\"{e}\", 
                            ExternalPredictor cannot be created"""
            logger.error(exception_msg)
            raise ValueError(exception_msg)

    def get_prediction(self, payload: Dict) -> FMBenchPredictionResponse:
        # Represents the prompt payload
        prompt_input_data = payload['inputs']
        # Initialize the key in the inference spec for each model. If the key is 
        # an OpenAI API key or a Gemini API key, it will be initialized here.
        # The inference format for each option (OpenAI/Gemini) is the same using LiteLLM
        # for streaming/non-streaming
        # set the environment for the specific model 
        if 'gemini' in self.endpoint_name:
            os.environ["GEMINI_API_KEY"] = self._api_key
        else:
            os.environ["OPENAI_API_KEY"] = self._api_key
        latency: Optional[float] = None
        completion_tokens: Optional[int] = None
        prompt_tokens: Optional[int] = None
        response_dict_from_streaming: Optional[Dict] = None
        TTFT: Optional[float] = None
        TPOT: Optional[float] = None
        TTLT: Optional[float] = None

        try:
            # recording latency for when streaming is enabled
            st = time.perf_counter()
            response = completion(model=self._endpoint_name,
                                messages=[{"content": prompt_input_data,
                                            "role": "user"}],
                                    temperature=self._temperature,
                                    max_tokens=self._max_tokens,
                                    caching=self._caching,
                                    stream=self._stream)
            logger.info(f"stop token: {self._stop}, streaming: {self._stream}, "
                        f"response: {response}")

            # Get the response and the TTFT, TPOT, TTLT metrics if the streaming
            # for responses is set to true
            if self._stream is True:
                response_dict_from_streaming = get_response_stream(response,
                                                                   st,
                                                                   self._start,
                                                                   self._stop,
                                                                   is_sagemaker=False)
                TTFT = response_dict_from_streaming.get('TTFT')
                TPOT = response_dict_from_streaming.get('TPOT')
                TTLT = response_dict_from_streaming.get('TTLT')
                response = response_dict_from_streaming['response']
                self._response_json["generated_text"] = json.loads(response)[0].get('generated_text')
                # Getting in the total input and output tokens using token counter.
                # Streaming on liteLLM does not support prompt tokens and completion tokens 
                # in the invocation response format
                prompt_tokens = token_counter(model=self._endpoint_name,
                                              messages=[{"content": prompt_input_data,
                                                         "role": "user"}])
                completion_tokens = token_counter(text=self._response_json["generated_text"])
                # Extract latency in seconds
                latency = time.perf_counter() - st
                logger.info(f"streaming prompt token count: {prompt_tokens}, "
                            f"completion token count: {completion_tokens}, latency: {latency}")
            # If streaming is set to false, then get the response in the normal
            # without streaming format from LiteLLM
            else:
                # Iterate through the entire model response
                # Since we are not sending batched requests so we only expect a single completion
                for choice in response.choices:
                    # Extract the message and the message's content from LiteLLM
                    if choice.message and choice.message.content:
                        # Extract the response from the dict
                        self._response_json["generated_text"] = choice.message.content
                        break
                # Extract number of input and completion prompt tokens
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                # Extract latency in seconds
                latency = response._response_ms / 1000
        except Exception as e:
            logger.error(f"exception during prediction, endpoint_name={self._endpoint_name}, "
                         f"exception={e}")
        return FMBenchPredictionResponse(response_json=self._response_json,
                                         latency=latency,
                                         time_to_first_token=TTFT,
                                         time_per_output_token=TPOT,
                                         time_to_last_token=TTLT,
                                         completion_tokens=completion_tokens,
                                         prompt_tokens=prompt_tokens)

    def calculate_cost(self,
                       instance_type: str,
                       instance_count: int,
                       pricing: Dict,
                       duration: float,
                       prompt_tokens: int,
                       completion_tokens: int) -> float:
        """Represents the function to calculate the cost for OpenAI/Gemini experiments.
        instance_type represents the model name
        """

        # Initializing all cost variables
        experiment_cost: Optional[float] = None
        input_token_cost: Optional[float] = None
        output_token_cost: Optional[float] = None
        try:
            if self._pt_model_id is None:
                logger.info("calculate_cost, calculating cost with token based pricing")
                # Retrieve the pricing information for the instance type
                non_aws_model_pricing = pricing['pricing']['token_based']
                # Calculate cost based on the number of input and output tokens
                model_pricing = non_aws_model_pricing.get(instance_type, None)
                if model_pricing:
                    input_token_cost = (prompt_tokens / 1000.0) * model_pricing['input-per-1k-tokens']
                    output_token_cost = (completion_tokens / 1000.0) * model_pricing['output-per-1k-tokens']
                    experiment_cost = input_token_cost + output_token_cost
                    logger.info(f"instance_type={instance_type}, prompt_tokens={prompt_tokens}, "
                                f"input_token_cost={input_token_cost}, output_token_cost={completion_tokens}, "
                                f"output_token_cost={output_token_cost}, experiment_cost={experiment_cost}")
                else:
                    logger.error(f"model pricing for \"{instance_type}\" not found, "
                                 f"cannot calculate experiment cost")
            else:
                logger.info("calculate_cost, calculating cost with PT pricing")
                instance_based_pricing = pricing['pricing']['instance_based']
                hourly_rate = instance_based_pricing.get(instance_type, None)
                # calculating the experiment cost for instance based pricing
                duration_in_hours_ceil = math.ceil(duration/3600)
                experiment_cost = hourly_rate * duration_in_hours_ceil
                logger.info(f"instance_type={instance_type}, hourly_rate={hourly_rate}, "
                            f"duration_in_hours_ceil={duration_in_hours_ceil}, experiment_cost={experiment_cost}")

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
    def endpoint_name(self) -> str:
        """The endpoint name property."""
        return self._endpoint_name

    @property
    def inference_parameters(self) -> Dict:
        """The inference parameters property."""
        return dict(temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    top_p=self._top_p)


def create_predictor(endpoint_name: str, inference_spec: Optional[Dict], metadata: Optional[Dict]):
    return ExternalPredictor(endpoint_name, inference_spec, metadata)