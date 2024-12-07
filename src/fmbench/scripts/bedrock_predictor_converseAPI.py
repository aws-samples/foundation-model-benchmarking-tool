import os
import math
import json
import time
import boto3
import logging
import pandas as pd
from datetime import datetime
from fmbench.scripts import constants
from typing import Dict, Optional, List
from botocore.exceptions import ClientError
from fmbench.scripts.stream_responses import get_response_stream
from fmbench.scripts.fmbench_predictor import (FMBenchPredictor,
                                             FMBenchPredictionResponse)

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the global variables
EMBEDDING_MODELS: List[str] = ["amazon.titan-embed-text-v1",
                             "cohere.embed-english-v3",
                             "cohere.embed-multilingual-v3"]

# This is the bedrock service name that is used to create
# the bedrock runtime client
BEDROCK_RUNTIME_CLIENT: str = 'bedrock-runtime'

class BedrockConversePredictor(FMBenchPredictor):
    def __init__(self,
                 endpoint_name: str,
                 inference_spec: Optional[Dict],
                 metadata: Optional[Dict]):
        try:
            self._endpoint_name = endpoint_name
            self._pt_model_id = None
            self._inference_spec = inference_spec
            self._aws_region = boto3.Session().region_name
            self._system_prompts = [{"text": "You are a helpful AI assistant."}]

            # Initialize the Bedrock client
            self._bedrock_client = boto3.client(BEDROCK_RUNTIME_CLIENT)

            # Check for provisioned throughput endpoint
            if ':provisioned-model/' in self._endpoint_name:
                logger.info(f"{self._endpoint_name} is a provisioned throughput endpoint")
                bedrock_client = boto3.client('bedrock')
                response = bedrock_client.list_provisioned_model_throughputs()
                if response['ResponseMetadata']['HTTPStatusCode'] != 200:
                    logger.error("Error in list_provisioned_model_throughputs")
                else:
                    fm_arn = [pt_summary['foundationModelArn'] for \
                             pt_summary in response['provisionedModelSummaries'] \
                             if pt_summary['provisionedModelArn'] == self._endpoint_name]
                    if len(fm_arn) > 0:
                        self._pt_model_id = self._endpoint_name
                        self._endpoint_name = fm_arn[0].split("/")[1]
                        logger.info(f"PT found, pt_model_id={self._pt_model_id}, endpoint_name={self._endpoint_name}")
                    else:
                        logger.error("No matching PT found")

            # Default inference parameters
            self._temperature = 0.1
            self._max_tokens = 100
            self._top_p = 0.9
            self._top_k = 200
            self._stream = None

            # Override defaults if inference spec is provided
            if inference_spec:
                parameters: Optional[Dict] = inference_spec.get('parameters')
                if parameters:
                    self._temperature = parameters.get('temperature', self._temperature)
                    self._max_tokens = parameters.get('max_tokens', self._max_tokens)
                    self._top_p = parameters.get('top_p', self._top_p)
                    self._top_k = parameters.get('top_k', self._top_k)
                    self._stream = inference_spec.get("stream", self._stream)
                    if 'system_prompt' in parameters:
                        self._system_prompts = [{"text": parameters['system_prompt']}]
            self._response_json = {}
            logger.info(f"Initialized BedrockConversePredictor with model={self._endpoint_name}")
        except Exception as e:
            exception_msg = f"Exception while creating predictor: {str(e)}"
            logger.error(exception_msg)
            raise ValueError(exception_msg)

    def get_prediction(self, payload: Dict) -> FMBenchPredictionResponse:
        prompt_input_data = payload['inputs']
        base64_img = payload.get('base64_img')
        latency = None
        completion_tokens = None
        prompt_tokens = None
        TTFT = TPOT = TTLT = None

        # Prepare messages format
        if base64_img is not None:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_input_data},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}" if not base64_img.startswith('data:image/') else base64_img
                        }
                    }
                ]
            }]
        else:
            messages = [{
                "role": "user",
                "content": [{"text": prompt_input_data}]
            }]

        # Prepare inference configuration
        inference_config = {
            "temperature": self._temperature,
            "maxTokens": self._max_tokens,
            "topP": self._top_p,
        }

        # Implement retry logic
        INITIAL_RETRY_DELAY = 2.0
        MAX_RETRY_DELAY = 60.0
        retry_count = 0

        while True:
            try:
                st = time.perf_counter()
                response = self._bedrock_client.converse(
                    modelId=self._endpoint_name,
                    messages=messages,
                    system=self._system_prompts,
                    inferenceConfig=inference_config
                )

                # Extract response and token usage
                self._response_json["generated_text"] = response['output']['message']['content'][0]['text']
                prompt_tokens = response['usage']['inputTokens']
                completion_tokens = response['usage']['outputTokens']
                latency = time.perf_counter() - st

                logger.info(f"Response generated. Input tokens: {prompt_tokens}, "
                          f"Output tokens: {completion_tokens}, Latency: {latency}")
                break

            except ClientError as e:
                if e.response['Error']['Code'] in ['ThrottlingException', 'TooManyRequestsException']:
                    retry_count += 1
                    wait_time = min(INITIAL_RETRY_DELAY * (2 ** (retry_count - 1)), MAX_RETRY_DELAY)
                    logger.warning(f"Throttling error. Retrying in {wait_time:.2f} seconds... (Attempt {retry_count})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"ClientError: {str(e)}")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise

        return FMBenchPredictionResponse(
            response_json=self._response_json,
            latency=latency,
            time_to_first_token=TTFT,
            time_per_output_token=TPOT,
            time_to_last_token=TTLT,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens
        )

    def calculate_cost(self,
                       instance_type: str,
                       instance_count: int,
                       pricing: Dict,
                       duration: float,
                       prompt_tokens: int,
                       completion_tokens: int) -> float:
        """Represents the function to calculate the cost for Bedrock experiments.
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
                bedrock_pricing = pricing['pricing']['token_based']
                # Calculate cost based on the number of input and output tokens
                model_pricing = bedrock_pricing.get(instance_type, None)
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

    def shutdown(self) -> None:
        """Represents the function to shutdown the predictor
           cleanup the endpooint/container/other resources
        """
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

    @property
    def platform_type(self) -> Dict:
        """The inference parameters property."""
        return constants.PLATFORM_BEDROCK

def create_predictor(endpoint_name: str, inference_spec: Optional[Dict], metadata: Optional[Dict]):
    if endpoint_name in EMBEDDING_MODELS:
        logger.error(f"embeddings models not supported for now")
        return None
    else:
        return BedrockConversePredictor(endpoint_name, inference_spec, metadata)