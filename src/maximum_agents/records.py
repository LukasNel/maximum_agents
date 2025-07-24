from pydantic import BaseModel
from enum import Enum
from typing import Union, Optional, Dict, Any

class PartType(str, Enum):
    THINKING = "thinking"
    CODE = "code"
    OUTPUT = "output"
    TOOL_CALL = "tool_call"
class ThinkingPartT(BaseModel):
    type: PartType = PartType.THINKING
    content: str

class CodePartT(BaseModel):
    type: PartType = PartType.CODE
    content: str

class OutputPartT(BaseModel):
    type: PartType = PartType.OUTPUT
    content: str

class ToolCallT(BaseModel):
    type: PartType = PartType.TOOL_CALL
    name: str
    arguments: Dict[str, Any]
    
    def to_str(self) -> str:
        return f"Tool: {self.name}\nArguments: {self.arguments}"

class OutputType(str, Enum):
    BASIC = "basic"

class BasicAnswerT(BaseModel):
    answer: str

class ResultT[T: BaseModel](BaseModel):
    output: OutputType = OutputType.BASIC
    answer: T

PartT = Union[ThinkingPartT, CodePartT, OutputPartT, ToolCallT]

class StepT(BaseModel):
    step_number: Optional[int] = None
    parts: list[PartT]