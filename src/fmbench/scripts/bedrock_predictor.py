import os
import copy
import boto3
import logging
from litellm import embedding
from litellm import completion 
from typing import Dict, Optional
from fmbench.scripts.fmbench_predictor import FMBenchPredictor, FMBenchPredictionResponse

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the global variables, including the embeddings model declaration and the service name
global_vars = globals()
global_vars['EMBEDDING_MODELS']: list[str] = ["amazon.titan-embed-text-v1", "cohere.embed-english-v3", "cohere.embed-multilingual-v3"]  # type: ignore
global_vars['SERVICE_NAME']: str = 'bedrock'

# Represents the class to predict using a bedrock rest API 
class BedrockPredictor(FMBenchPredictor):

    # initialize the service name
    service_name = global_vars['SERVICE_NAME']

    # overriding abstract method
    def __init__(self, endpoint_name: str, inference_spec: Dict | None):
        try:  
            self._endpoint_name = endpoint_name
            self._inference_spec = inference_spec
            self._predictor = boto3.client('bedrock-runtime')
            self.aws_region = boto3.Session().region_name
            # this is used to invoke the litellm API with the specific bedrock model offering of choice
            self.bedrock_model = f"{self.service_name}/{self.endpoint_name}"
            self.response_json = {}
            logger.info(f"__init__ self._predictor={self._predictor}")
        except Exception as e:
            logger.error(f"Exception occurred while creating predictor/initializing variables for endpoint_name={self._endpoint_name}, exception={e}")
            self._predictor = None

    def get_prediction(self, payload: Dict) -> FMBenchPredictionResponse:
        # Represents the prompt payload
        prompt_input_data = payload['inputs']
        # Represents the inference parameters (in this case, temperature and caching) 
        parameters = copy.deepcopy(payload['parameters'])

        # get the temperature, max_tokens and caching values as inference parameters 
        temperature = parameters.get('temperature', None)
        max_tokens = parameters.get('max_tokens', None)
        caching = parameters.get('caching', None)
        logger.info(f"Temperature passed for bedrock invocation: {temperature}, Max tokens: {max_tokens}, Caching: {caching}... ")

        os.environ["AWS_REGION_NAME"] = self.aws_region
        try:
            # this response is for text generation models on bedrock: CLAUDE, LLAMA, ai21, MISTRAL, MIXTRAL, COHERE
            logger.info(f"Invoking {self.bedrock_model} to get inference....")
            response = completion(
            model=self.bedrock_model,
            temperature = temperature,
            max_tokens = max_tokens,
            caching = caching,
            messages=[{ "content": prompt_input_data,"role": "user"}], 
            )
            # iterate through the entire model response
            for choice in response.choices:
                # extract the message and the message's content from litellm
                if choice.message and choice.message.content:
                    # extract the response from the dict
                    self.response_json["generated_text"] = choice.message.content
                    break

            # Extract number of input and completion prompt tokens
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            # Extract latency in seconds
            latency_ms = response._response_ms
            latency = latency_ms / 1000
        except Exception as e:
            logger.error(f"Exception occurred during prediction for endpoint_name={self._endpoint_name}, exception={e}")
        return FMBenchPredictionResponse(response_json=self.response_json, latency=latency, completion_tokens=completion_tokens, prompt_tokens=prompt_tokens)

    def calculate_cost(self, instance_type: str, config: dict, duration: float, metrics: dict) -> float:
        """Represents the function to calculate the cost for Bedrock experiments."""
        # Initializing all cost variables
        experiment_cost: Optional[float] = 0.0  
        input_token_cost: Optional[float] = 0.0
        output_token_cost: Optional[float] = 0.0
        try:
            if metrics:
                prompt_tokens = metrics.get("all_prompts_token_count", None)
                completion_tokens = metrics.get("all_completions_token_count", None)
                # Retrieve the pricing information for the instance type
                instance_pricing = config['pricing'].get(instance_type, []) 
                logger.info(f"Instance pricing for {instance_type}: {instance_pricing}")
                # Calculate cost based on the number of input and output tokens
                for pricing in instance_pricing:
                    input_token_cost += (prompt_tokens / 1000.0) * pricing.get('input-per-1k-tokens', 0)
                    output_token_cost += (completion_tokens / 1000.0) * pricing.get('output-per-1k-tokens', 0)
                logger.info(f"input token cost: {input_token_cost}")
                logger.info(f"output token cost: {output_token_cost}")
                experiment_cost = input_token_cost + output_token_cost
        except Exception as e:
            logger.error(f"Exception occurred during experiment cost calculation....., exception={e}")
        return experiment_cost

    @property
    def endpoint_name(self) -> str:
        """The endpoint name property."""
        return self._endpoint_name
    
# Subclass of BedrockPredictor for embedding models supported on Amazon Bedrock
class BedrockPredictorEmbeddings(BedrockPredictor):
    def get_prediction(self, payload: Dict) -> FMBenchPredictionResponse:
        # Represents the prompt payload
        prompt_input_data = payload['inputs'] 
        # Represents the inference parameters (in this case, temperature and caching) 
        parameters = copy.deepcopy(payload['parameters'])
        # get the temperature, max_tokens and caching values as inference parameters 
        temperature = parameters.get('temperature', None)
        max_tokens = parameters.get('max_tokens', None)
        caching = parameters.get('caching', None)
        os.environ["AWS_REGION_NAME"] = self.aws_region
        try:
            logger.info(f"Invoking {self.bedrock_model} embeddings model to get inference....")
            response = embedding(
                model=self.bedrock_model,
                temperature = temperature,
                max_tokens = max_tokens,
                caching = caching,
                input=[prompt_input_data],
            )
            embedding_vector = response.data[0]["embedding"]
            self.response_json["generated_text"] = str(embedding_vector)
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.total_tokens 
            latency_ms = response._response_ms
            latency = latency_ms / 1000 
        except Exception as e:
            logger.error(f"Exception occurred during prediction for endpoint_name={self._endpoint_name}, exception={e}")
        return FMBenchPredictionResponse(response_json=self.response_json, latency=latency, completion_tokens=completion_tokens, prompt_tokens=prompt_tokens)

def create_predictor(endpoint_name: str, inference_spec: Dict | None):
    if endpoint_name in global_vars['EMBEDDING_MODELS']: 
        return BedrockPredictorEmbeddings(endpoint_name, inference_spec) 
    else:
        return BedrockPredictor(endpoint_name, inference_spec)