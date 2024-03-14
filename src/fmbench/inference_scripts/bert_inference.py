import time
import json
import boto3
import logging
from fmbench.utils import *

# Assuming the logger is already configured elsewhere in your application
logger = logging.getLogger(__name__)

# Helper function to query the BERT endpoint with encoded text
def query_endpoint(predictor, encoded_text, inference_params):
    # Assume inference_params include ContentType and Accept headers
    response = predictor.predict(
        encoded_text, inference_params
    )
    return response

# Function to parse the model's response
def parse_response(response):
    if isinstance(response, bytes):
        response = response.decode('utf-8')
    response_json = json.loads(response)
    return response_json

def set_metrics(endpoint_name=None, prompt=None, inference_params=None, prompt_tokens=None,
                completion_tokens=None, probabilities=None, labels=None, completion=None, latency=None) -> Dict:
    # Adjusting the metrics dictionary to fit the BERT model output
    return dict(endpoint_name=endpoint_name, prompt=prompt, **inference_params, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, probabilities=probabilities, labels=labels,
                completion=completion, latency=latency)

def get_inference(predictor, payload) -> Dict:
    latency = 0
    try:
        prompt_tokens = count_tokens(payload['inputs'])
        logger.info(f"get_inference, endpoint={predictor.endpoint_name}, prompt_tokens={prompt_tokens}")

        # Starting the timer for latency 
        st = time.perf_counter()
        response = query_endpoint(predictor, payload['inputs'], payload['parameters'])
        latency = time.perf_counter() - st

        response_json = parse_response(response)

        # Extract necessary information from the response
        probabilities = response_json.get("probabilities", [])
        labels = response_json.get("labels", [])
        completion = response_json.get("predicted_label")
        completion_tokens = count_tokens(completion)  

        response = set_metrics(endpoint_name=predictor.endpoint_name, prompt=payload['inputs'],
                               inference_params=payload['parameters'], prompt_tokens=prompt_tokens,
                               completion_tokens=completion_tokens, probabilities=probabilities, labels=labels,
                               completion=completion, latency=latency)

        logger.info(f"get_inference, done, endpoint={predictor.endpoint_name}, completion={completion}, latency={latency:.2f}")
    except Exception as e:
        logger.error(f"Error occurred with {predictor.endpoint_name}, exception={str(e)}")
        response = set_metrics(endpoint_name=predictor.endpoint_name, prompt=payload['inputs'],
                               inference_params=payload['parameters'], prompt_tokens=None, completion_tokens=None,
                               probabilities=None, labels=None, completion=None, latency=None)
    return response


