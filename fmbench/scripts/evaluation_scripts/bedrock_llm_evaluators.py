import math
import time
import json
import boto3
import logging
from datetime import datetime
from typing import Dict, Optional
from botocore.exceptions import ClientError
from litellm import completion, token_counter, RateLimitError
from fmbench.scripts.fmbench_predictor import (FMBenchEvaluation,
                                               FMBenchEvaluationResponse)
from fmbench.scripts.evaluation_scripts.bedrock_predictor_converseAPI import invoke_bedrock_converse


# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the global variables, including the embeddings model declaration
# and the service name
EMBEDDING_MODELS: List[str] = ["amazon.titan-embed-text-v1",
                               "cohere.embed-english-v3",
                               "cohere.embed-multilingual-v3"]
SERVICE_NAME: str = 'bedrock'

class BedrockEvaluation(FMBenchEvaluation):
    """
    An implementation of FMBenchEvaluation that uses a BedrockPredictor.
    It implements:
      - get_llm_evaluation: calls the predictor to generate an answer and gathers token usage info.
      - calculate_llm_eval_cost: computes cost based on token usage (or hourly pricing if using a PT model).
    """
    def __init__(self,
                 endpoint_name: str,
                 inference_spec: Optional[Dict],
                 metadata: Optional[Dict]):
        self._endpoint_name = endpoint_name
        self._pt_model_id = None
        self._inference_spec = inference_spec
        self._aws_region = boto3.Session().region_name

    def get_llm_evaluation(self, payload: Dict) -> FMBenchEvaluationResponse:
        """
        Uses the underlying bedrock model id to get a prediction from the model.
        Then extracts the response JSON, token counts, and computes cost.
        """
        logger.info(f"Getting evaluation for payload: {payload}")
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
                # cohere does not support top_p and apprarently LiteLLM does not
                # know that?
                if 'cohere' not in self._endpoint_name:
                    logger.info(f"Invoking {self._bedrock_model} to get inference")
                    st = time.perf_counter()
                    response = completion(model=self._bedrock_model,
                                        model_id=self._pt_model_id,
                                        messages=messages,
                                        temperature=self._temperature,
                                        max_tokens=self._max_tokens,
                                        top_p=self._top_p,
                                        caching=self._caching,
                                        stream=self._stream)
                    # Extract latency in seconds
                    latency = time.perf_counter() - st
                else:
                    logger.info(f"Invoking {self._bedrock_model} to get inference")
                    st = time.perf_counter()
                    response = completion(model=self._bedrock_model,
                                        model_id=self._pt_model_id,
                                        messages=messages,
                                        temperature=self._temperature,
                                        max_tokens=self._max_tokens,
                                        caching=self._caching,
                                        stream=self._stream)
                    latency = time.perf_counter() - st
                    logger.info(f"stop token: {self._stop}, streaming: {self._stream}, "
                                f"response: {response}")
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
                # Calculate the total cost (would need implementation based on your pricing model)
                total_cost = self.calculate_cost(prompt_tokens, completion_tokens)
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

        return FMBenchEvaluationResponse(
                    response_json=self._response_json,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_cost=total_cost,
                    latency=latency)

    def calculate_llm_eval_cost(self,
                                instance_type: str,
                                instance_count: int,
                                config: Dict,
                                duration: float,
                                metrics: Dict) -> float:
        """
        Computes the evaluation cost.
        If the predictor was created with a provisioned throughput model (_pt_model_id is set),
        then use instance-based pricing. Otherwise, use token-based pricing.
        """
        experiment_cost: Optional[float] = None
        try:
            # Check if we are using token-based pricing.
            if self._predictor._pt_model_id is None:
                logger.info("Calculating cost using token-based pricing")
                token_pricing = self._pricing.get("token_based", {})
                model_pricing = token_pricing.get(instance_type)
                if model_pricing:
                    input_token_cost = (metrics["prompt_tokens"] / 1000.0) * model_pricing.get("input-per-1k-tokens", 0)
                    output_token_cost = (metrics["completion_tokens"] / 1000.0) * model_pricing.get("output-per-1k-tokens", 0)
                    experiment_cost = input_token_cost + output_token_cost
                    logger.info(f"instance_type={instance_type}, prompt_tokens={metrics['prompt_tokens']}, "
                                f"input_token_cost={input_token_cost}, completion_tokens={metrics['completion_tokens']}, "
                                f"output_token_cost={output_token_cost}, experiment_cost={experiment_cost}")
                else:
                    logger.error(f"Model pricing for {instance_type} not found in token_based pricing")
            else:
                # Otherwise, we assume the model is provisioned and use instance-based pricing.
                logger.info("Calculating cost using instance-based pricing")
                instance_pricing = self._pricing.get("instance_based", {})
                hourly_rate = instance_pricing.get(instance_type)
                if hourly_rate is not None:
                    # Convert duration from seconds to hours and round up.
                    duration_hours = math.ceil(duration / 3600)
                    experiment_cost = hourly_rate * duration_hours
                    logger.info(f"instance_type={instance_type}, hourly_rate={hourly_rate}, "
                                f"duration_hours={duration_hours}, experiment_cost={experiment_cost}")
                else:
                    logger.error(f"Hourly pricing for {instance_type} not found in instance_based pricing")
        except Exception as e:
            logger.error(f"Exception during cost calculation: {e}")
        return experiment_cost if experiment_cost is not None else 0.0