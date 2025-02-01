import time
import boto3
import logging
from typing import Dict, Optional

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BEDROCK_RUNTIME: str = "bedrock-runtime"

def invoke_bedrock_converse(
    endpoint_name: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    top_p: float,
    system_prompts: list = [{"text": "You are a helpful AI assistant."}]
) -> Dict:
    """
    Simple function to invoke Bedrock's converse API.
    Args:
        endpoint_name: The name of the Bedrock endpoint
        messages: List of message dictionaries
        temperature: Temperature parameter for inference
        max_tokens: Maximum tokens to generate
        top_p: Top-p parameter for inference
        system_prompts: System prompts to use (default provided)
    Returns:
        Dict containing response data
    """
    response: Optional[Dict] = None
    bedrock_client = boto3.client(BEDROCK_RUNTIME)
    inference_config = {
        "temperature": temperature,
        "maxTokens": max_tokens,
        "topP": top_p,
    }
    st = time.perf_counter()
    response = bedrock_client.converse(
        modelId=endpoint_name,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config
    )
    latency = time.perf_counter() - st
    return response, latency
