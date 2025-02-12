import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from abc import ABC, abstractmethod, abstractproperty


class FMBenchEvaluation(ABC):
    """
    This base class is used during the evaluation process by FMBench. It contains the following 
    two methods:
    1. get_llm_evaluation: This function evaluates the model under test based on ground truth provided in the dataset.
                           This function keeps track of the prompt and completion tokens, and the input/output token
                           cost.
    2. calculate_llm_eval_cost: This function calculates the evaluation price based on the model. If the model is hosted
                                on an instance that has hourly pricing, the function uses the hourly price to determine
                                the evaluation cost. If the model has a token based pricing, then the input and output
                                tokens are used to determine the cost of evaluation.
    """
    @abstractmethod
    def __init__(self,
                 endpoint_name: str,
                 inference_spec: Optional[Dict],
                 metadata: Optional[Dict]):
        pass

    @abstractmethod
    def get_llm_evaluation(self, payload: Dict) -> Dict:
        pass

    @abstractmethod
    def calculate_llm_eval_cost(self,
                       instance_type: str,
                       instance_count: int,
                       config: Dict,
                       duration: float,
                       metrics: Dict) -> float:
        """Represents the function to calculate the
           cost of each experiment run.
        """
        pass


class FMBenchEvaluationResponse(dict):
    def __init__(self, *k, **kwargs):
        self.__dict__ = self
        self.__dict__['response_json'] = kwargs['response_json']
        self.__dict__['prompt_tokens'] = kwargs['prompt_tokens']
        self.__dict__['completion_tokens'] = kwargs['completion_tokens']
        self.__dict__['total_cost'] = kwargs['total_cost']
        super().__init__(*k, **kwargs)
