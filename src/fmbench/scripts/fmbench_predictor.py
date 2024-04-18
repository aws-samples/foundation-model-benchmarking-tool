from typing import Dict, Optional
from abc import ABC, abstractmethod, abstractproperty


class FMBenchPredictor(ABC):

    @abstractmethod
    def __init__(self,
                 endpoint_name: str,
                 inference_spec: Optional[Dict]):
        pass

    @abstractmethod
    def get_prediction(self, payload: Dict) -> Dict:
        pass

    @abstractmethod
    def calculate_cost(self,
                       instance_type: str,
                       config: Dict,
                       duration: float,
                       metrics: Dict) -> float:
        """Represents the function to calculate the
           cost of each experiment run.
        """
        pass

    @abstractproperty
    def endpoint_name(self) -> str:
        """The endpoint name property."""
        pass

    @abstractproperty
    def inference_parameters(self) -> Dict:
        """The inference parameters property."""
        pass


class FMBenchPredictionResponse(dict):

    def __init__(self, *k, **kwargs):
        self.__dict__ = self
        self.__dict__['response_json'] = kwargs['response_json']
        self.__dict__['latency'] = kwargs['latency']
        self.__dict__['prompt_tokens'] = kwargs['prompt_tokens']
        self.__dict__['completion_tokens'] = kwargs['completion_tokens']
        super().__init__(*k, **kwargs)