from typing import Dict
from abc import ABC, abstractmethod


class FMBenchPredictor(ABC):
    
    @abstractmethod
    def __init__(self, endpoint_name: str):
        pass
    @abstractmethod
    def get_prediction(self, payload: Dict) -> Dict:
        pass