from typing import Dict
from abc import ABC, abstractmethod, abstractproperty
class FMBenchPredictor(ABC):
    
    @abstractmethod
    def __init__(self, endpoint_name: str, inference_spec: Dict | None):
        pass
    @abstractmethod
    def get_prediction(self, payload: Dict) -> Dict:
        pass
    @abstractproperty
    def endpoint_name(self) -> str:
        """The endpoint name property."""
        pass
    
class FMBenchPredictionResponse(dict):
   def __init__(self, *k, **kwargs):
      self.__dict__ = self
      self.__dict__['response_json'] = kwargs['response_json']
      self.__dict__['latency'] = kwargs['latency']
      super().__init__(*k, **kwargs)
