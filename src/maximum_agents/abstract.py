from abc import ABC, abstractmethod
from typing import Callable
from smolagents import Tool
from .records import  ResultT, StepT

class AbstractAgent(ABC):
    @abstractmethod
    def run(self, task: str, log: Callable[[StepT], None]) -> ResultT:
        raise NotImplementedError("Subclasses must implement this method")



