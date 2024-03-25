import os
import boto3
import logging
from typing import Dict
from litellm import embedding
from litellm import completion
from typing import Dict, List, Tuple
from fmbench.scripts.fmbench_predictor import FMBenchPredictor, FMBenchPredictionResponse

## set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


## Represents the class to predict using a bedrock rest API 
class BedrockPredictor(FMBenchPredictor):

    # overriding abstract method
    def __init__(self, endpoint_name: str, inference_spec: Dict):

        ## initiliazing the bedrock client
        bedrock_client = boto3.client('bedrock-runtime')

        self._predictor: Optional[bedrock_client] = None
        self._endpoint_name: str = endpoint_name
        self._inference_spec = inference_spec

        try:
            # Create a bedrock runtime client 
            self._predictor = bedrock_client

        except Exception as e:
            logger.error(f"create_predictor, exception occured while creating predictor for endpoint_name={self._endpoint_name}, exception={e}")
        logger.info(f"__init__ self._predictor={self._predictor}")
        
    def get_prediction(self, payload: Dict) -> FMBenchPredictionResponse:
        response_json = {}
        latency = None
        response = None
        prompt_tokens = None
        completion_tokens = None

        ## Represents the prompt payload
        prompt_input_data = payload['inputs']

        # Get the AWS region dynamically
        aws_region = boto3.Session().region_name

        ## getting the aws account region as an environment variable as declared in the litellm documentation
        os.environ["AWS_REGION_NAME"] = aws_region

        # Build the REST API URL based on the region and self.endpoint_name
        bedrock_model = f"bedrock/{self.endpoint_name}"
        logger.info(f"Endpoint name being invoked for inference using 'litellm': {bedrock_model}")

        ## Represents calling the litellm completion/messaging api utilizing the completion/embeddings API
        ## [CLAUDE, LLAMA, ai21, MISTRAL, MIXTRAL, COHERE]
        logger.info(f"Invoking {bedrock_model} to get inference....")
        response = completion(
        model=bedrock_model,
        messages=[{ "content": prompt_input_data,"role": "user"}], 
        )

        ## iterate through the entire model response
        for choice in response.choices:
            ## extract the message and the message's content from litellm
            if choice.message and choice.message.content:
                ## extract the response from the dict
                response_json["generated_text"] = choice.message.content
                logger.info(f"Inference from {bedrock_model}: {response_json}")
                break

        # Extract number of input and completion prompt tokens (this is the same structure for embeddings and text generation models on Amazon Bedrock)
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        # Extract latency in seconds
        latency_ms = response._response_ms
        latency = latency_ms / 1000

        return FMBenchPredictionResponse(response_json=response_json, latency=latency, completion_tokens=completion_tokens, prompt_tokens=prompt_tokens)

    def calculate_cost(self, instance_type, config: dict, duration: float, metrics: dict) -> float:
        """Represents the function to calculate the cost for Bedrock experiments."""
        experiment_cost = 0.0

        if metrics:
            prompt_tokens = metrics.get("all_prompts_token_count", 0)
            completion_tokens = metrics.get("all_completions_token_count", 0)

            # Retrieve the pricing information for the instance type
            instance_pricing = config['pricing'].get(instance_type, {})
            input_token_price_per_thousand = 0
            output_token_price_per_thousand = 0

            for pricing_dict in instance_pricing:
                logger.info(f"pricing dict: {pricing_dict}")
                if 'input-per-1k-tokens' in pricing_dict:
                    input_token_price_per_thousand = pricing_dict['input-per-1k-tokens']
                    logger.info(f"input per 1k token pricing: {input_token_price_per_thousand}")
                if 'output-per-1k-tokens' in pricing_dict:
                    output_token_price_per_thousand = pricing_dict['output-per-1k-tokens']
                    logger.info(f"output per 1k token pricing: {output_token_price_per_thousand}")

            # Calculate cost based on the number of input and output tokens
            input_token_cost = (prompt_tokens / 1000.0) * input_token_price_per_thousand
            output_token_cost = (completion_tokens / 1000.0) * output_token_price_per_thousand

            experiment_cost = input_token_cost + output_token_cost

        return experiment_cost

    @property
    def endpoint_name(self) -> str:
        """The endpoint name property."""
        return self._endpoint_name
    
class BedrockPredictorEmbeddings(BedrockPredictor):
    def get_prediction(self, payload: Dict) -> FMBenchPredictionResponse:
        response_json = {}
        latency = None
        response = None
        prompt_tokens = None
        completion_tokens = None

        ## Represents the prompt payload
        prompt_input_data = payload['inputs']

        # Get the AWS region dynamically
        aws_region = boto3.Session().region_name

        ## getting the aws account region as an environment variable as declared in the litellm documentation
        os.environ["AWS_REGION_NAME"] = aws_region

        # Build the REST API URL based on the region and self.endpoint_name
        bedrock_model = f"bedrock/{self.endpoint_name}"
        logger.info(f"Endpoint name being invoked for inference using 'litellm': {bedrock_model}")

        ## Represents calling the litellm completion/messaging api utilizing the completion/embeddings API
        ## [CLAUDE, LLAMA, ai21, MISTRAL, MIXTRAL, COHERE]
        logger.info(f"Invoking {bedrock_model} to get inference....")
        response = embedding(
            model=bedrock_model,
            input=[prompt_input_data],
        )

        embedding_vector = response.data[0]["embedding"]
        response_json["generated_text"] = str(embedding_vector)
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.total_tokens

        latency_ms = response._response_ms
        latency = latency_ms / 1000

        return FMBenchPredictionResponse(response_json=response_json, latency=latency, completion_tokens=completion_tokens, prompt_tokens=prompt_tokens)


def create_predictor(endpoint_name: str, inference_spec: Dict):
    ## represents the current bedrock embedding models
    embedding_models = ["amazon.titan-embed-text-v1", "cohere.embed-english-v3", "cohere.embed-multilingual-v3"]
    if endpoint_name in embedding_models: ## handling for embedding models
        return BedrockPredictorEmbeddings(endpoint_name, inference_spec)
    else: ## handling for all text generation models
        return BedrockPredictor(endpoint_name, inference_spec)