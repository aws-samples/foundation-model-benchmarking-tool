import time
import json
import copy
import boto3
import logging
import sagemaker
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from fmbench.scripts import constants
from fmbench.utils import count_tokens
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from fmbench.scripts.stream_responses import get_response_stream
from fmbench.scripts.sagemaker_metrics import get_endpoint_metrics
from fmbench.scripts.fmbench_predictor import (FMBenchPredictor,
                                               FMBenchPredictionResponse)

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the SageMaker runtime to support getting response streams
sagemaker_runtime = boto3.client('sagemaker-runtime')
sm_client = boto3.client("sagemaker")


class SageMakerPredictor(FMBenchPredictor):
    # overriding abstract method
    def __init__(self,
                 endpoint_name: str,
                 inference_spec: Optional[Dict],
                 metadata: Optional[Dict]):
        self._predictor: Optional[sagemaker.base_predictor.Predictor] = None
        self._endpoint_name: str = endpoint_name
        self._inference_spec: Dict = inference_spec
        self._variant_name: Optional[str] = None
        self._use_messages_api_format: Optional[bool] = None
        if metadata is not None:
            self._variant_name = metadata.get("variant_name")
            self._use_messages_api_format = metadata.get("use_messages_api_format")

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
                    f"_variant_name={self._variant_name}_"
                    f"inference_spec={self._inference_spec}, "
                    f"_use_messages_api_format={self._use_messages_api_format}")

    def get_prediction(self, payload: Dict) -> FMBenchPredictionResponse:
        response_json: Optional[Dict] = None
        response: Optional[str] = None
        latency: Optional[float] = None
        TTFT: Optional[float] = None
        TPOT: Optional[float] = None
        TTLT: Optional[float] = None
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        streaming: Optional[bool] = None
        stop_token: Optional[str] = None
        
        
        
        # represents the number of tokens in the prompt payload
        prompt_tokens = count_tokens(payload["inputs"])

        try:
            container_type = self._inference_spec.get("container_type")
            st = time.perf_counter()
            split_input_and_inference_params = None
            if self._inference_spec is not None:
                split_input_and_inference_params = self._inference_spec.get("split_input_and_parameters")
            response = None
            streaming = self._inference_spec.get("stream", False)
            if split_input_and_inference_params is True:
                response = self._predictor.predict(payload["inputs"],
                                                   self._inference_spec["parameters"])
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
                    payload = payload | dict(self._inference_spec["parameters"])
                else:
                    # the final payload should look like this:
                    # {
                    #   "parameters": { 
                    #     "top_p": 0.9,
                    #     "max_tokens": 100
                    #    },
                    #   "inputs": "this is the prompt"
                    # }
                    if container_type == constants.CONTAINER_TYPE_HUGGINGFACE:
                        payload2 = copy.deepcopy(payload)
                        payload2['text_inputs'] = payload2.pop('inputs')
                        payload2['mode'] = "embedding"
                    else:
                        payload = payload | dict(parameters=self._inference_spec["parameters"])

            # if the response streaming is step, call the get_response stream on the 
            # sagemaker endpoint, else use the simple predict call
            if streaming is True:
                start_token = self._inference_spec.get("start_token", None)
                stop_token = self._inference_spec.get("stop_token", None)
                payload["stream"] = streaming
                logger.info(f"streaming={streaming}, calling invoke_endpoint_with_response_stream")
                response_stream = sagemaker_runtime.invoke_endpoint_with_response_stream(
                                                    EndpointName=self._endpoint_name,
                                                    Body=json.dumps(payload),
                                                    ContentType="application/json")
                response_dict = get_response_stream(response_stream['Body'],
                                                    st,
                                                    start_token,
                                                    stop_token,
                                                    is_sagemaker=True)
                TTFT = response_dict.get('TTFT')
                TPOT = response_dict.get('TPOT')
                TTLT = response_dict.get('TTLT')
                response = response_dict.get('response')
            else:
                logger.info(f"streaming={streaming}, calling predict")
                if container_type:
                    response = self._predictor.predict(payload2)
                    logger.info("Running predictor with HuggingFace Container for Embedding model")
                else:
                    response = self._predictor.predict(payload)
                    logger.info("Running predictor for Foundation model")

            latency = time.perf_counter() - st
            if isinstance(response, bytes):
                response = response.decode('utf-8')
            response_json = json.loads(response)

            # we want to set the "generated_text" key in the response
            if isinstance(response_json, list):
                response_json = response_json[0]
                # add a key called completion, if not there
                if response_json.get("generated_text") is None:
                    # look for predicted label and set that as generated text
                    if response_json.get("predicted_label") is not None:
                        response_json["generated_text"] = response_json.get("predicted_label")
                    else:
                        logger.error("response_json is list but did not contain generated_text or predicted_label, dont know how to handle this")
            elif isinstance(response_json, dict):
                choices = response_json.get("choices")
                if choices is not None:
                    if isinstance(choices, list):
                        response_json = response_json["choices"][0]["message"]
                        if response_json.get("generated_text") is None:
                            if response_json.get("content") is not None:
                                response_json["generated_text"] = response_json.get("content")
                            else:
                                logger.error(f"response_json is a dict, choices is a list, but response_json does not contain generated_text, dont know how to handle this")
                        else:
                            logger.error(f"response_json is a dict, choices is a list, but generated_text ia None, dont know how to handle this")
                    else:
                        logger.error(f"response_json is a dict, but choices is not a list but rather it is {type(choices)}, dont know how to handle this")
                else:
                    if container_type == constants.CONTAINER_TYPE_HUGGINGFACE:
                        response_json["generated_text"] = response_json.get("embedding")
                        completion_tokens = len(response_json.get("generated_text"))
                    # logger.error(f"response_json is a dict, but does not contain choices, dont know how to handle this")
            else:
                logger.error(f"response_json data type is {type(response_json)}, dont know how to handle this")
            # counts the completion tokens for the model using the default/user provided tokenizer
            if not completion_tokens:
                completion_tokens = count_tokens(response_json.get("generated_text"))

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

    @property
    def endpoint_name(self) -> str:
        """The endpoint name property."""
        return self._endpoint_name

    def calculate_cost(self,
                       instance_type: str,
                       instance_count: int,
                       pricing: Dict,
                       duration: float,
                       prompt_tokens: int,
                       completion_tokens: int) -> float:
        """Calculate the cost of each experiment run."""
        experiment_cost: Optional[float] = None
        try:
            instance_based_pricing = pricing['pricing']['instance_based']
            hourly_rate = instance_based_pricing.get(instance_type, None)
            logger.info(f"the hourly rate for running on {instance_type} is {hourly_rate}, "
                        f"instance_count={instance_count}")
            # calculating the experiment cost for instance based pricing
            instance_count = instance_count if instance_count else 1
            experiment_cost = (hourly_rate / 3600) * duration * instance_count
        except Exception as e:
            logger.error(f"exception occurred during experiment cost calculation, exception={e}")
        return experiment_cost

    def get_metrics(self,
                    start_time: datetime,
                    end_time: datetime,
                    period: int = 60) -> pd.DataFrame:
        return get_endpoint_metrics(self._endpoint_name, self._variant_name, start_time, end_time)
        
    def shutdown(self) -> None:
        """Represents the function to shutdown the predictor
           cleanup the endpooint/container/other resources
        """
        try:
            ep_name = self.endpoint_name
            ## Describe the model endpoint 
            logger.info(f"Going to describe the endpoint -> {ep_name}")
            resp = sm_client.describe_endpoint(EndpointName=ep_name)

            ## If the given model endpoint is in service, delete it 
            if resp['EndpointStatus'] == 'InService':
                logger.info(f"going to delete {ep_name}")
                ## deleting the model endpoint
                sm_client.delete_endpoint(EndpointName=ep_name)
                logger.info(f"deleted {ep_name}")
                return True
        except Exception as e:
            logger.error(f"error deleting endpoint={ep_name}, exception={e}")
            return False
    
    @property
    def inference_parameters(self) -> Dict:
        """The inference parameters property."""
        return self._inference_spec.get("parameters")


    @property
    def platform_type(self) -> Dict:
        """The inference parameters property."""
        return constants.PLATFORM_SAGEMAKER
    
def create_predictor(endpoint_name: str, inference_spec: Optional[Dict], metadata: Optional[Dict]):
    return SageMakerPredictor(endpoint_name, inference_spec, metadata)
