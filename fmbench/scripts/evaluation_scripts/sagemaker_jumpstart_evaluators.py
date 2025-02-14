import os
import re
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
        self._variant_name: Optional[str] = None
        self._use_messages_api_format: Optional[bool] = None
        if metadata is not None:
            self._variant_name = metadata.get("variant_name")
            self._use_messages_api_format = metadata.get("use_messages_api_format")
        # initialize a sagemaker predictor
        try:
            # Create a SageMaker Predictor object
            self._predictor = Predictor(
                endpoint_name=self._endpoint_name,
                sagemaker_session=sagemaker.Session(), 
                serializer=JSONSerializer()
            )
            logger.info(f"Initialized the sagemaker predictor for {endpoint_name}")
        except Exception as e:
            logger.error(f"create_predictor, exception occured while creating predictor "
                         f"for endpoint_name={self._endpoint_name}, exception={e}")
        logger.info(f"__init__ _predictor={self._predictor}, "
                    f"_variant_name={self._variant_name}_"
                    f"inference_spec={self._inference_spec}, "
                    f"_use_messages_api_format={self._use_messages_api_format}")

    def get_llm_evaluation(self, model_id: str, prompt: str) -> FMBenchEvaluationResponse:
        """
        Uses the underlying SageMaker predictor to get a prediction from the model.
        Then extracts the response, token counts, and latency.
        """
        logger.info(f"Getting evaluation for prompt: {prompt}")
        try:
            # initialize the completion
            llm_completion: Optional[str] = None
            # Initialize the payload dictionary
            payload: Dict={}
            # Create a payload for the SageMaker Jumpstart model endpoint
            payload['inputs']=prompt
            # print(f"self._inference_spec DEBUG: {self._inference_spec}")
            payload = payload | dict(parameters=self._inference_spec["parameter_set"])
            split_input_and_inference_params = None
            if self._inference_spec is not None:
                split_input_and_inference_params = self._inference_spec.get("split_input_and_parameters")
            response = None
            if split_input_and_inference_params is True:
                response = self._predictor.predict(payload["inputs"],
                                                   self._inference_spec["parameter_set"])
            else:
                if self._use_messages_api_format is True:
                    # if needed in future add support for system prompt as well
                    # and multiple system/user prompts but all that is not needed for now
                    payload = {"messages": [{"role": "user",
                                             "content": payload["inputs"]}]}
                    # the final payload should look like this:
                    # {
                    #   "top_p": 0.9,
                    #   "max_tokens": 100,
                    #   "messages": [
                    #     {
                    #       "role": "user",
                    #       "content": "this is the prompt"
                    #     }
                    #   ]
                    # }
                    payload = payload | dict(self._inference_spec["parameter_set"])
                else:
                    payload = payload | dict(parameters=self._inference_spec["parameter_set"])
            logger.info(f"PAYLOAD FORMAT DEBUG: {payload}")
            st = time.perf_counter()
            # Pass the serialized payload (bytes) to the predictor
            response = self._predictor.predict(payload)
            latency = time.perf_counter() - st
            if isinstance(response, bytes):
                response = response.decode('utf-8')
            response_json = json.loads(response)
            logger.info(f"response json: {response_json}")
            prompt_tokens = count_tokens(prompt)
            logger.info(f"prompt tokens: {prompt_tokens}")
            response_json = response_json[0]
            llm_completion = response_json.get("generated_text")
            completion_tokens = count_tokens(llm_completion)
            # This regex matches the first occurrence of a substring that starts with '{' and ends with '}'.
            # The DOTALL flag ensures that newlines are included in the match. This is done to extract out the json
            # that the LLM evaluator is supposed to return
            pattern = r'(\{.*\})'
            match = re.search(pattern, llm_completion, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in the provided text.")
            llm_completion = match.group(1)
            logger.info(f"Response from the llm judge: {llm_completion}")
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
    
def create_evaluator(endpoint_name: str, inference_spec: Optional[Dict]):
    return SageMakerEvaluation(endpoint_name, inference_spec)