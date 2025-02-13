import os
import math
import time
import json
import boto3
import logging
from datetime import datetime
from typing import Dict, Optional, List
from botocore.exceptions import ClientError
from litellm import completion, token_counter, RateLimitError
from fmbench.scripts.evaluation_scripts.fmbench_evaluator import (FMBenchEvaluation,
                                               FMBenchEvaluationResponse)


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
    # initialize the service name
    _service_name: str = SERVICE_NAME
    
    def __init__(self,
                 endpoint_name: str, 
                 inference_spec: Optional[Dict] = None):
        self._endpoint_name = endpoint_name
        self._pt_model_id = None
        self._inference_spec = inference_spec
        self._aws_region = boto3.Session().region_name
        self._bedrock_model = f"{self._service_name}/{self._endpoint_name}"
        # litellm supports the following inference params as per
        # https://litellm.vercel.app/docs/completion/input
        self._temperature = 0.1
        self._max_tokens = 100
        self._top_p = 0.9
        # no caching of responses since we want every inference
        # call to be independant
        self._caching = False

    def get_llm_evaluation(self, model_id: str, prompt: str) -> FMBenchEvaluationResponse:
        """
        Uses the underlying bedrock model id to get a prediction from the model.
        Then extracts the response JSON, token counts, and computes cost.
        """
        logger.info(f"Getting evaluation for prompt: {prompt}")
        os.environ["AWS_REGION_NAME"] = self._aws_region
        latency: Optional[float] = None
        completion_tokens: Optional[int] = None
        prompt_tokens: Optional[int] = None
        llm_completion: Optional[str] = None
        
        # add logic to retry if there are throttling errors
        INITIAL_RETRY_DELAY: float = 2.0 
        MAX_RETRY_DELAY: float = 60.0  
        retry_count = 0
        
        while True:
            try:
                messages = [{"content": prompt, "role": "user"}]
                logger.info(f"Invoking {self._bedrock_model} to get inference")
                kwargs = {
                    "model": self._bedrock_model,
                    "model_id": model_id,
                    "messages": messages,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                    "caching": self._caching
                }
                if 'cohere' not in self._endpoint_name:
                    kwargs["top_p"] = self._top_p
                st = time.perf_counter()
                response = completion(**kwargs)
                latency = time.perf_counter() - st
                # Process response
                if hasattr(response, 'choices') and response.choices:
                    for choice in response.choices:
                        if hasattr(choice, 'message') and choice.message and choice.message.content:
                            llm_completion = choice.message.content
                            break
                if hasattr(response, 'usage'):
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                if hasattr(response, '_response_ms'):
                    latency = response._response_ms / 1000
                break
            except (RateLimitError, ClientError) as e:
                # Retry logic for throttling errors
                if isinstance(e, ClientError) and e.response['Error']['Code'] not in ['ThrottlingException', 'TooManyRequestsException', 'ServiceUnavailableException']:
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
            llm_completion=llm_completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency=latency)

    def calculate_llm_eval_cost(
        self,
        instance_type: str,
        instance_count: int,
        pricing: Dict,
        duration: float,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        experiment_cost: float = 0.0
        try:
            if self._pt_model_id is None:
                # Use token-based pricing.
                logger.info("calculate_cost: using token-based pricing")
                token_pricing = pricing.get("pricing", {}).get("token_based", {})
                model_pricing = token_pricing.get(instance_type)
                if model_pricing:
                    input_token_cost = (prompt_tokens / 1000.0) * model_pricing.get("input-per-1k-tokens", 0)
                    output_token_cost = (completion_tokens / 1000.0) * model_pricing.get("output-per-1k-tokens", 0)
                    experiment_cost = input_token_cost + output_token_cost
                    logger.info(
                        f"instance_type={instance_type}, prompt_tokens={prompt_tokens}, "
                        f"input_token_cost={input_token_cost}, completion_tokens={completion_tokens}, "
                        f"output_token_cost={output_token_cost}, experiment_cost={experiment_cost}"
                    )
                else:
                    logger.error(f"Model pricing for '{instance_type}' not found in token_based pricing.")
            else:
                # Use instance-based (hourly) pricing.
                logger.info("calculate_cost: using instance-based pricing")
                instance_pricing = pricing.get("pricing", {}).get("instance_based", {})
                hourly_rate = instance_pricing.get(instance_type)
                if hourly_rate is not None:
                    # Convert duration from seconds to hours, rounding up.
                    duration_hours = math.ceil(duration / 3600)
                    # Multiply by instance_count if more than one instance is used.
                    experiment_cost = hourly_rate * duration_hours * instance_count
                    logger.info(
                        f"instance_type={instance_type}, hourly_rate={hourly_rate}, "
                        f"duration_hours={duration_hours}, instance_count={instance_count}, "
                        f"experiment_cost={experiment_cost}"
                    )
                else:
                    logger.error(f"Hourly pricing for '{instance_type}' not found in instance_based pricing.")
        except Exception as e:
            logger.error(f"Exception during cost calculation: {e}")
        return experiment_cost
    
def create_evaluator(endpoint_name: str, inference_spec: Optional[Dict]):
    return BedrockEvaluation(endpoint_name)