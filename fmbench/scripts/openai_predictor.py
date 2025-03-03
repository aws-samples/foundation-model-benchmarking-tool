import os
import json
import copy
import time
import stat
import logging
import litellm
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from fmbench.scripts import constants
from fmbench.utils import count_tokens
from fmbench.scripts.inference_containers.utils import get_accelerator_type
from fmbench.scripts.fmbench_predictor import (FMBenchPredictor,
                                             FMBenchPredictionResponse)

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIPredictor(FMBenchPredictor):
    """OpenAI Predictor class for FMBench.

    This class implements the FMBenchPredictor interface for multiple model platforms
    using LiteLLM as the unified interface. It supports:
    - OpenAI API models (e.g., gpt-3.5-turbo, gpt-4)
    - Ollama models (e.g., ollama/llama2, ollama/mistral)
    - vLLM served models (through their API endpoint)

    The predictor automatically configures itself using API keys from either:
    1. Environment variables (OPENAI_API_KEY for OpenAI)
    2. The /tmp/fmbench-read file
    
    For Ollama and vLLM, you need to specify the full model path including the platform prefix:
    - Ollama models: "ollama/modelname" (e.g., "ollama/llama2")
    - vLLM models: "vllm/modelname@endpoint" (e.g., "vllm/mistral@http://localhost:8000")

    Attributes:
        _endpoint_name (str): The full model identifier including platform prefix
        _inference_spec (Dict): Specification for inference parameters like temperature, max_tokens
        _accelerator (str): Type of accelerator (used for local deployments)
        _platform (str): The platform being used (openai, ollama, or vllm)
    """

    def __init__(self,
                 endpoint_name: str,
                 inference_spec: Optional[Dict], 
                 metadata: Optional[Dict]):
        """Initialize the unified predictor.

        Args:
            endpoint_name (str): Full model identifier (e.g., "ollama/llama2", "vllm/mistral@http://localhost:8000")
            inference_spec (Optional[Dict]): Dictionary containing inference parameters
                like temperature, max_tokens, etc.
            metadata (Optional[Dict]): Additional metadata (not used for setup)

        Raises:
            ValueError: If API key cannot be found for OpenAI or if model identifier is invalid
            Exception: For other initialization errors
        """
        try:
            self._endpoint_name: str = endpoint_name
            self._inference_spec: Dict = inference_spec
            self._accelerator = get_accelerator_type()
            # Start collecting EC2 metrics. This will be called once.
            collect_ec2_metrics()
            
        except Exception as e:
            logger.error(f"create_predictor, exception occurred while creating predictor "
                        f"for endpoint_name={endpoint_name}, exception={e}")
        logger.info(f"_endpoint_name={self._endpoint_name}, _inference_spec={self._inference_spec}")

    def get_prediction(self, payload: Dict) -> FMBenchPredictionResponse:
        """Get a prediction from the model.

        Makes an API call using litellm to get a model completion.
        Tracks various metrics like latency and token counts.

        Args:
            payload (Dict): Dictionary containing the input prompt under 'inputs' key

        Returns:
            FMBenchPredictionResponse: Object containing:
                - response_json: Dictionary with generated text
                - latency: Total API call time in seconds
                - time_to_first_token: Not implemented for API-based models
                - time_per_output_token: Not implemented for API-based models
                - time_to_last_token: Not implemented for API-based models
                - completion_tokens: Number of tokens in the response
                - prompt_tokens: Number of tokens in the input
        """
        response_json: Optional[Dict] = None
        response: Optional[str] = None
        latency: Optional[float] = None
        TTFT: Optional[float] = None
        TPOT: Optional[float] = None
        TTLT: Optional[float] = None
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        container_type: Optional[str] = None
        # Get the prompt
        prompt: str = payload['inputs']
        prompt_tokens = count_tokens(payload['inputs'])

        
        if container_type == constants.CONTAINER_TYPE_OPENAI:
                self._platform = 'openai'
                self._model_name = endpoint_name
                # Try to get OpenAI API key from environment variable first
                api_key = os.environ.get("OPENAI_API_KEY")
                # If not in environment, try to read from file
                if not api_key:
                    api_key_path = Path("/tmp/fmbench-read/openai_api_key.txt")
                    if api_key_path.exists():
                        api_key = api_key_path.read_text().strip()
                
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment or /tmp/fmbench-read")
                    
                litellm.api_key = api_key
            else:
                # Any other endpoint is basically platform + model name concatenated together
                self._model_name = container_type + "/" + endpoint_name
        try:
            # Prepare the payload
            split_input_and_inference_params: Optional[bool] = None
            if self._inference_spec is not None:
                split_input_and_inference_params = self._inference_spec.get("split_input_and_parameters")
                container_type = self._inference_spec.get("container_type", constants.CONTAINER_TYPE_DJL)
                logger.debug(f"split input parameters is: {split_input_and_inference_params}, "
                             f"container_type={container_type}")
            # Add any additional parameters from inference_spec
            params = self._inference_spec.get("parameters", {})
            
            # Start latency timer
            st = time.perf_counter()
            
            # Make the API call using litellm
            # For Ollama and vLLM, we use the specific model format
            model = (
                f"ollama/{self._model_name}" if self._platform == 'ollama'
                else f"vllm/{self._model_name}" if self._platform == 'vllm'
                else self._model_name
            )
            
            response = litellm.completion(
                model=model,
                messages=messages,
                **params
            )
            
            # Record latency
            latency = time.perf_counter() - st

            # Extract the generated text
            generated_text = response.choices[0].message.content
            response_json = {"generated_text": generated_text}

            # Get token counts from the response
            # Note: Some platforms might not provide token counts
            try:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
            except AttributeError:
                # If token counts aren't available, try to estimate them
                prompt_tokens = count_tokens(prompt)
                completion_tokens = count_tokens(generated_text)

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
        """Get the endpoint name.

        Returns:
            str: The name of the OpenAI model being used
        """
        return self._endpoint_name

    def calculate_cost(self,
                      instance_type: str,
                      instance_count: int,
                      pricing: Dict,
                      duration: float,
                      prompt_tokens: int,
                      completion_tokens: int) -> float:
        """Calculate the cost of the API call based on token usage.

        OpenAI charges based on the number of tokens used, with different rates for
        input (prompt) and output (completion) tokens. The rates vary by model.

        Args:
            instance_type (str): Not used for OpenAI
            instance_count (int): Not used for OpenAI
            pricing (Dict): Dictionary containing token-based pricing information
            duration (float): Not used for OpenAI
            prompt_tokens (int): Number of tokens in the input prompt
            completion_tokens (int): Number of tokens in the model's response

        Returns:
            float: Total cost in USD, or None if calculation fails
        """
        experiment_cost: Optional[float] = None
        try:
            token_based_pricing = pricing['pricing']['token_based']
            model_pricing = token_based_pricing.get(self._endpoint_name, {})
            
            input_cost_per_1k = model_pricing.get('input_cost_per_1k', 0)
            output_cost_per_1k = model_pricing.get('output_cost_per_1k', 0)
            
            input_cost = (prompt_tokens / 1000) * input_cost_per_1k
            output_cost = (completion_tokens / 1000) * output_cost_per_1k
            
            experiment_cost = input_cost + output_cost
            
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
        """Get the inference parameters.

        Returns:
            Dict: Dictionary of parameters used for inference (temperature, max_tokens, etc.)
        """
        return self._inference_spec.get("parameters")
    
    @property
    def platform_type(self) -> Dict:
        """The inference parameters property."""
        return constants.PLATFORM_EC2


def create_predictor(endpoint_name: str, inference_spec: Optional[Dict], metadata: Optional[Dict]):
    """Create an OpenAI predictor instance.

    Factory function to create and return a configured OpenAI predictor.

    Args:
        endpoint_name (str): Name of the OpenAI model to use
        inference_spec (Optional[Dict]): Dictionary of inference parameters
        metadata (Optional[Dict]): Additional metadata (not used)

    Returns:
        OpenAIPredictor: Configured predictor instance
    """
    return OpenAIPredictor(endpoint_name, inference_spec, metadata) 