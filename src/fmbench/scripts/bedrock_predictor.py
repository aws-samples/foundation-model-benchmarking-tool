import os
import boto3
import logging
from litellm import completion
from typing import Dict, Optional, List
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
    def __init__(self, endpoint_name: str, inference_spec: Optional[Dict]):
        try:
            # initialize private member variables
            self._endpoint_name = endpoint_name
            self._inference_spec = inference_spec
            self._predictor = boto3.client('bedrock-runtime')
            self._aws_region = boto3.Session().region_name

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
            self._stop = None

            # no caching of responses since we want every inference
            # call to be independant
            self._caching = False

            self._response_json = {}
            logger.info(f"__init__, _bedrock_model={self._bedrock_model}, "
                        f"_predictor={self._predictor}")
        except Exception as e:
            logger.error(f"exception while creating predictor/initializing variables "
                         f"for endpoint_name={self._endpoint_name}, exception={e}")
            self._predictor = None

    def get_prediction(self, payload: Dict) -> FMBenchPredictionResponse:
        # Represents the prompt payload
        prompt_input_data = payload['inputs']
        # Represents the inference parameters (in this case, temperature and caching) 
        parameters = payload['parameters']

        # get the temperature, max_tokens and caching values as inference parameters 
        temperature = parameters.get('temperature', self._temperature)
        max_tokens = parameters.get('max_tokens', self._max_tokens)
        top_p = parameters.get('top_p', self._top_p)
        caching = parameters.get('caching', self._caching)
        logger.info(f"parameters={parameters}")

        os.environ["AWS_REGION_NAME"] = self._aws_region
        latency: Optional[float] = None
        completion_tokens: Optional[int] = None
        prompt_tokens: Optional[int] = None

        try:
            # this response is for text generation models on bedrock: Claude, Llama, Mistral etc.
            logger.info(f"Invoking {self._bedrock_model} to get inference")
            # cohere does not support top_p and apprarently LiteLLM does not
            # know that?
            if 'cohere' not in self._endpoint_name:
                response = completion(model=self._bedrock_model,
                                      messages=[{"content": prompt_input_data,
                                                 "role": "user"}],
                                      temperature=temperature,
                                      max_tokens=max_tokens,
                                      top_p=top_p,
                                      caching=caching,)
            else:
                response = completion(model=self._bedrock_model,
                                      messages=[{"content": prompt_input_data,
                                                 "role": "user"}],
                                      temperature=temperature,
                                      max_tokens=max_tokens,
                                      caching=caching,)

            # iterate through the entire model response
            # since we are not sending bached request so we only expect
            # a single completion
            for choice in response.choices:
                # extract the message and the message's content from litellm
                if choice.message and choice.message.content:
                    # extract the response from the dict
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
                                         completion_tokens=completion_tokens,
                                         prompt_tokens=prompt_tokens)

    def calculate_cost(self,
                       instance_type: str,
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

        except Exception as e:
            logger.error(f"exception occurred during experiment cost calculation, exception={e}")
        return experiment_cost

    @property
    def endpoint_name(self) -> str:
        """The endpoint name property."""
        return self._endpoint_name


def create_predictor(endpoint_name: str, inference_spec: Optional[Dict]):
    if endpoint_name in EMBEDDING_MODELS:
        logger.error(f"embeddings models not supported for now")
        return None
    else:
        return BedrockPredictor(endpoint_name, inference_spec)