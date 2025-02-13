import os
import time
import json
import boto3
import logging
import sagemaker
from datetime import datetime
from typing import Dict, Optional, List
from fmbench.utils import count_tokens
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from fmbench.scripts.evaluation_scripts.fmbench_evaluator import (FMBenchEvaluation,
                                               FMBenchEvaluationResponse)

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SageMakerEvaluation(FMBenchEvaluation):
    """
    An implementation of FMBenchEvaluation that uses a SageMakerPredictor.
    It implements:
      - get_llm_evaluation: calls the predictor to generate an answer and gathers token usage info.
      - calculate_llm_eval_cost: computes cost based on token usage or instance pricing.
    """
    def __init__(self,
                 endpoint_name: str,
                 inference_spec: Optional[Dict] = None,
                 metadata: Optional[Dict] = None):
        self._predictor: Optional[sagemaker.base_predictor.Predictor] = None
        self._endpoint_name = endpoint_name
        self._inference_spec = inference_spec
        self._metadata = metadata
        # initialize a sagemaker predictor
        try:
            # Create a SageMaker Predictor object
            self._predictor = Predictor(
                endpoint_name=self._endpoint_name,
                sagemaker_session=sagemaker.Session(),
                serializer=JSONSerializer()
            )
        except Exception as e:
            logger.error(f"create_predictor, exception occured while creating predictor "
                         f"for endpoint_name={self._endpoint_name}, exception={e}")
        logger.info(f"__init__ _predictor={self._predictor}, "
                    f"inference_spec={self._inference_spec} ")

    def get_llm_evaluation(self, model_id: str, prompt: str) -> FMBenchEvaluationResponse:
        """
        Uses the underlying SageMaker predictor to get a prediction from the model.
        Then extracts the response, token counts, and latency.
        """
        logger.info(f"Getting evaluation for prompt: {prompt}")
        try:
            # initialize the completion
            llm_completion: Optional[str] = None
            st = time.perf_counter()
            prediction_response = self._predictor.predict(prompt)
            latency = time.perf_counter() - st if prediction_response.latency is None else prediction_response.latency
            if isinstance(response, bytes):
                response = response.decode('utf-8')
            response_json = json.loads(response)
            prompt_tokens = count_tokens(prompt)
            # we want to set the "generated_text" key in the response
            if isinstance(response_json, list):
                response_json = response_json[0]
                # add a key called completion, if not there
                if response_json.get("generated_text") is None:
                    # look for predicted label and set that as generated text
                    if response_json.get("predicted_label") is not None:
                        llm_completion = response_json.get("predicted_label")
                    else:
                        logger.error("response_json is list but did not contain generated_text or predicted_label, dont know how to handle this")
            elif isinstance(response_json, dict):
                choices = response_json.get("choices")
                if choices is not None:
                    if isinstance(choices, list):
                        response_json = response_json["choices"][0]["message"]
                        if response_json.get("generated_text") is None:
                            if response_json.get("content") is not None:
                                llm_completion = response_json.get("content")
                            else:
                                logger.error(f"response_json is a dict, choices is a list, but response_json does not contain generated_text, dont know how to handle this")
                        else:
                            logger.error(f"response_json is a dict, choices is a list, but generated_text ia None, dont know how to handle this")
                    else:
                        logger.error(f"response_json is a dict, but choices is not a list but rather it is {type(choices)}, dont know how to handle this")
            completion_tokens = count_tokens(llm_completion)
        except Exception as e:
            logger.error(f"Error during SageMaker evaluation, endpoint_name={self._endpoint_name}, "
                        f"exception={e}")
            raise
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
        """
        Calculate the cost of evaluation based on either token-based or instance-based pricing.
        """
        experiment_cost: float = 0.0
        try:
            # Check if token-based pricing is available for this model
            token_pricing = pricing.get("pricing", {}).get("token_based", {})
            model_pricing = token_pricing.get(instance_type)
            
            if model_pricing:
                # Use token-based pricing
                logger.info("calculate_cost: using token-based pricing")
                input_token_cost = (prompt_tokens / 1000.0) * model_pricing.get("input-per-1k-tokens", 0)
                output_token_cost = (completion_tokens / 1000.0) * model_pricing.get("output-per-1k-tokens", 0)
                experiment_cost = input_token_cost + output_token_cost
                logger.info(
                    f"instance_type={instance_type}, prompt_tokens={prompt_tokens}, "
                    f"input_token_cost={input_token_cost}, completion_tokens={completion_tokens}, "
                    f"output_token_cost={output_token_cost}, experiment_cost={experiment_cost}"
                )
            else:
                # Use instance-based (hourly) pricing
                logger.info("calculate_cost: using instance-based pricing")
                instance_pricing = pricing.get("pricing", {}).get("instance_based", {})
                hourly_rate = instance_pricing.get(instance_type)
                if hourly_rate is not None:
                    # Convert duration from seconds to hours and calculate cost
                    duration_hours = duration / 3600
                    experiment_cost = hourly_rate * duration_hours * instance_count
                    logger.info(
                        f"instance_type={instance_type}, hourly_rate={hourly_rate}, "
                        f"duration_hours={duration_hours}, instance_count={instance_count}, "
                        f"experiment_cost={experiment_cost}"
                    )
                else:
                    logger.error(f"Pricing for '{instance_type}' not found in either token_based or instance_based pricing.")
        except Exception as e:
            logger.error(f"Exception during cost calculation: {e}")
        return experiment_cost
    
def create_evaluator(endpoint_name: str):
    return SageMakerEvaluation(endpoint_name)