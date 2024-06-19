import time
import json
import logging
import sagemaker
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from fmbench.utils import count_tokens
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from fmbench.scripts.sagemaker_metrics import get_endpoint_metrics
from fmbench.scripts.fmbench_predictor import (FMBenchPredictor,
                                               FMBenchPredictionResponse)

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None

        # represents the number of tokens in the prompt payload
        prompt_tokens = count_tokens(payload["inputs"])

        try:
            st = time.perf_counter()
            split_input_and_inference_params = None
            if self._inference_spec is not None:
                split_input_and_inference_params = self._inference_spec.get("split_input_and_parameters")
            response = None
            response = None
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
                    payload = payload | dict(parameters=self._inference_spec["parameters"])
                #import json
                #logger.info(json.dumps(payload, indent=2, default=str))
                response = self._predictor.predict(payload)

            latency = time.perf_counter() - st
            if isinstance(response, bytes):
                response = response.decode('utf-8')
            response_json = json.loads(response)

            if isinstance(response_json, list):
                response_json = response_json[0]
            # add a key called completion, if not there
            if response_json.get("generated_text") is None:
                if response_json.get("predicted_label") is not None:
                    response_json["generated_text"] = response_json.get("predicted_label")
            # counts the completion tokens for the model using the default/user provided tokenizer
            completion_tokens = count_tokens(response_json.get("generated_text"))

        except Exception as e:
            logger.error(f"get_prediction, exception occurred while getting prediction for payload={payload} "
                         f"from predictor={self._endpoint_name}, response={response}, exception={e}")
        return FMBenchPredictionResponse(response_json=response_json,
                                         latency=latency,
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
        
    @property
    def inference_parameters(self) -> Dict:
        """The inference parameters property."""
        return self._inference_spec.get("parameters")


def create_predictor(endpoint_name: str, inference_spec: Optional[Dict], metadata: Optional[Dict]):
    return SageMakerPredictor(endpoint_name, inference_spec, metadata)
