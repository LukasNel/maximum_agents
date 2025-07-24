from pydantic import BaseModel
from enum import Enum
from typing import Union, Optional

class PartType(Enum, str):
    THINKING = "thinking"
    CODE = "code"
    OUTPUT = "output"

class ThinkingPartT(BaseModel):
    type: PartType = PartType.THINKING
    content: str

class CodePartT(BaseModel):
    type: PartType = PartType.CODE
    content: str

class OutputPartT(BaseModel):
    type: PartType = PartType.OUTPUT
    content: str

class OutputType(Enum, str):
    BASIC = "basic"

class BasicAnswerT(BaseModel):
    answer: str

class ResultT[T: BaseModel](BaseModel):
    output: OutputType = OutputType.BASIC
    answer: T

PartT = Union[ThinkingPartT, CodePartT, OutputPartT]

class StepT(BaseModel):
    step_number: Optional[int] = None
    parts: list[PartT]