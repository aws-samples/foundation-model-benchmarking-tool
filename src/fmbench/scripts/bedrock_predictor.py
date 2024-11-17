import os
import math
import json
import time
import boto3
import litellm
import logging
import pandas as pd
from datetime import datetime
from fmbench.scripts import constants
from typing import Dict, Optional, List
from botocore.exceptions import ClientError
from litellm import completion, token_counter
from litellm import completion, RateLimitError
from fmbench.scripts.stream_responses import get_response_stream
from fmbench.scripts.fmbench_predictor import (FMBenchPredictor,
                                               FMBenchPredictionResponse)


# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the global variables, including the embeddings model declaration
# and the service name
EMBEDDING_MODELS: List[str] = ["amazon.titan-embed-text-v1",
                               "cohere.embed-english-v3",
                               "cohere.embed-multilingual-v3"]
SERVICE_NAME: str = 'bedrock'


class BedrockPredictor(FMBenchPredictor):

    # initialize the service name
    _service_name: str = SERVICE_NAME

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

            # check if the endpoint name corresponded to a provisioned throughput
            # endpoint
            if ':provisioned-model/' in self._endpoint_name:
                logger.info(f"{self._endpoint_name} is a provisioned throughput endpoint")
                bedrock_client = boto3.client(SERVICE_NAME)
                response = bedrock_client.list_provisioned_model_throughputs()
                if response['ResponseMetadata']['HTTPStatusCode'] != 200:
                    logger.error(f"error received while calling list_provisioned_model_throughputs, response=\"{response}\", "
                                 f"BedrockPredictor cannot be created")
                else:
                    fm_arn = [pt_summary['foundationModelArn'] for \
                                pt_summary in response['provisionedModelSummaries'] \
                                  if pt_summary['provisionedModelArn'] == self._endpoint_name]
                    if len(fm_arn) > 0:                        
                        # set the PT name which looks like arn:aws:bedrock:us-east-1:<account-id>:provisioned-model/<something>
                        self._pt_model_id = self._endpoint_name
                        # set the endpoint name which needs to look like the FM model id
                        # this can now be extracted from the fm_arn which looks like 
                        # 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0:28k',
                        self._endpoint_name = fm_arn[0].split("/")[1]
                        logger.info(f"a matching PT was found, self._pt_model_id={self._pt_model_id}, "
                                    f"self._endpoint_name={self._endpoint_name}")
                    else:
                        logger.error(f"no matching PT was found, BedrockPredictor cannot be created")

            # model_id for the litellm API with the specific bedrock model of choice
            # endpoint_name in case of bedrock refers to the model_id such as 
            # cohere.command-text-v14 for example
            self._bedrock_model = f"{self._service_name}/{self._endpoint_name}"
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
                if parameters:
                    self._temperature = parameters.get('temperature', self._temperature)
                    self._max_tokens = parameters.get('max_tokens', self._max_tokens)
                    self._top_p = parameters.get('top_p', self._top_p)
                    self._stream = inference_spec.get("stream", self._stream)
                    self._stop = inference_spec.get("stop_token", self._stop)
                    self._start = inference_spec.get("start_token", self._start)
            self._response_json = {}
            logger.info(f"__init__, _bedrock_model={self._bedrock_model}, self._pt_model_id={self._pt_model_id},"
                        f"_temperature={self._temperature} "
                        f"_max_tokens={self._max_tokens}, _top_p={self._top_p} "
                        f"_stream={self._stream}, _stop={self._stop}, _caching={self._caching}")
        except Exception as e:
            exception_msg = f"""exception while creating predictor/initializing variables
                            for endpoint_name={self._endpoint_name}, exception=\"{e}\", 
                            BedrockPredictor cannot be created"""
            logger.error(exception_msg)
            raise ValueError(exception_msg)

    def get_prediction(self, payload: Dict) -> FMBenchPredictionResponse:
        # Represents the prompt payload
        prompt_input_data = payload['inputs']
        base64_img = payload.get('base64_img')
        os.environ["AWS_REGION_NAME"] = self._aws_region
        latency: Optional[float] = None
        completion_tokens: Optional[int] = None
        prompt_tokens: Optional[int] = None
        response_dict_from_streaming: Optional[Dict] = None
        TTFT: Optional[float] = None
        TPOT: Optional[float] = None
        TTLT: Optional[float] = None
        # add logic to retry if there are throttling errors
        INITIAL_RETRY_DELAY: float = 2.0 
        MAX_RETRY_DELAY: float = 60.0  
        retry_count = 0
        while True:
            try:
                # recording latency for when streaming is enabled
                st = time.perf_counter()
                logger.info(f"Invoking {self._bedrock_model} to get inference")

                # Get the base64 image if in vision mode
                if base64_img is not None:
                    logger.info("'base64_img' column provided in the dataset, going to use the multimodal"
                                "messages API to get inferences on the image")
                    # Add prefix if needed
                    if not base64_img.startswith('data:image/'):
                        base64_img = "data:image/jpeg;base64," + base64_img
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_input_data},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": base64_img
                                    },
                                },
                            ],
                        }
                    ]
                else:
                    logger.info("Going to use the standard text generation messages format to get inferences")
                    # Standard text generation format
                    messages = [{"content": prompt_input_data, "role": "user"}]
                # cohere does not support top_p and apprarently LiteLLM does not
                # know that?
                if 'cohere' not in self._endpoint_name:
                    response = completion(model=self._bedrock_model,
                                        model_id=self._pt_model_id,
                                        messages=messages,
                                        temperature=self._temperature,
                                        max_tokens=self._max_tokens,
                                        top_p=self._top_p,
                                        caching=self._caching,
                                        stream=self._stream)
                else:
                    response = completion(model=self._bedrock_model,
                                        model_id=self._pt_model_id,
                                        messages=messages,
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
                    prompt_tokens = token_counter(model=self._endpoint_name, messages=messages)
                    completion_tokens = token_counter(text=self._response_json["generated_text"])
                    # Extract latency in seconds
                    latency = time.perf_counter() - st
                    logger.info(f"streaming prompt token count: {prompt_tokens}, "
                                f"completion token count: {completion_tokens}, latency: {latency}")
                    logger.info("Completed streaming for the current UUID, moving to the next prediction.")
                    break  # Explicitly exit the loop for the current prediction
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
                    # If we get here, the call was successful, so we break out of the retry loop
                    break
            except (RateLimitError, ClientError) as e:
                # if the error is a throttling or too many requests exception, wait and retry again. The wait time between
                # each failed request increases exponentially
                if isinstance(e, ClientError) and e.response['Error']['Code'] not in ['ThrottlingException', 'TooManyRequestsException']:
                    logger.error(f"Unhandled ClientError: {str(e)}")
                    raise  # Re-raise if it's not a throttling error
                retry_count += 1
                wait_time = min(INITIAL_RETRY_DELAY * (2 ** (retry_count - 1)), MAX_RETRY_DELAY)
                logger.warning(f"Throttling error encountered: {str(e)}. Retrying in {wait_time:.2f} seconds... (Attempt {retry_count})")
                time.sleep(wait_time)

            except Exception as e:
                logger.error(f"Unexpected error during prediction, endpoint_name={self._endpoint_name}, "
                            f"exception={e}")
                raise  # Re-raise unexpected exceptions

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
        return BedrockPredictor(endpoint_name, inference_spec, metadata)
