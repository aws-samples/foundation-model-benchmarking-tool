import time
import json
import boto3
import logging
from fmbench.utils import *

# Assuming the logger is already configured elsewhere in your application
logger = logging.getLogger(__name__)

def set_metrics(endpoint_name=None,
                prompt=None,
                inference_params=None,
                completion=None,
                prompt_tokens=None,
                completion_tokens=None,
                latency=None) -> Dict:
    return dict(endpoint_name=endpoint_name,                
                prompt=prompt,
                **inference_params,
                completion=completion,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency=latency)

def get_inference(predictor, payload) -> Dict:
    smr_client = boto3.client("sagemaker-runtime")
    latency = 0

    try:
        prompt_tokens = count_tokens(payload['inputs'])
        logger.info(f"get_inference, endpoint={predictor.endpoint_name}, prompt_tokens={prompt_tokens}")

        # get inference
        st = time.perf_counter()        
        response = predictor.predict(payload)        
        latency = time.perf_counter() - st

        if isinstance(response, bytes):
            response = response.decode('utf-8')
        response_json = json.loads(response)
        if isinstance(response_json, list):
            response_json = response_json[0]

        completion = response_json.get("generated_text", "")
        completion_tokens = count_tokens(completion)

        response = set_metrics(predictor.endpoint_name,
                               payload['inputs'],
                               payload['parameters'],
                               completion,
                               prompt_tokens,
                               completion_tokens,
                               latency)
        logger.info(f"get_inference, done, endpoint={predictor.endpoint_name}, completion_tokens={completion_tokens}, latency={latency:.2f}")
    except Exception as e:
        logger.error(f"Error occurred with {predictor.endpoint_name}, exception={str(e)}")
        response = set_metrics(predictor.endpoint_name,
                               payload['inputs'],
                               payload['parameters'],
                               None,
                               prompt_tokens,
                               None,
                               None)

    return response
