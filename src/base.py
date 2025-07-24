from pydantic import BaseModel

from pydantic_final_answer_tools import PydanticFinalAnswerTool
from .abstract import AbstractAgent
from typing import Callable, Any, List, Dict, Optional, cast
from .records import  PartT, ResultT, BasicAnswerT, StepT, ThinkingPartT, CodePartT, OutputPartT
from smolagents import CodeAgent, Tool, LiteLLMModel, ChatMessage, ActionStep, PlanningStep, FinalAnswerStep, ChatMessageStreamDelta
from litellm.exceptions import InternalServerError
from .exponential_backoff import exponential_backoff_agentonly

class NoFinalResultError(Exception):
    pass


class RetryingModel(LiteLLMModel):
    @exponential_backoff_agentonly(
        max_retries=5, base_delay=1, max_delay=60, exceptions=(InternalServerError)
    )
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)


class CachedAnthropicModel(RetryingModel):
    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:

        new_messages_with_caching = []
        total_cache_limit = 4
        for message in reversed(messages):
            message = cast(Dict[str, Any], message)
            if isinstance(message["content"], str):
                new_message_copy: Dict[str, Any] = message.copy()
                content_block_new: Dict[str, Any] = {
                    "type": "text",
                    "text": message["content"],
                }
                if total_cache_limit > 0:
                    content_block_new["cache_control"] = {"type": "ephemeral"}
                    total_cache_limit -= 1
                new_message_copy["content"] = [content_block_new]
                new_messages_with_caching.append(new_message_copy)
            else:
                content_blocks_with_caching = []
                for content_block in message["content"]:
                    content_block_copy = content_block.copy()
                    if isinstance(content_block_copy, str):
                        if total_cache_limit > 0:
                            total_cache_limit -= 1
                            content_blocks_with_caching.append(
                                {
                                    "type": "text",
                                    "text": content_block_copy,
                                    "cache_control": {"type": "ephemeral"},
                                }
                            )
                        else:
                            content_blocks_with_caching.append(
                                {
                                    "type": "text",
                                    "text": content_block_copy,
                                }
                            )
                    else:
                        if total_cache_limit > 0:
                            total_cache_limit -= 1
                            content_block_copy["cache_control"] = {"type": "ephemeral"}
                        content_blocks_with_caching.append(content_block_copy)
                new_message_copy = message.copy()
                new_message_copy["content"] = content_blocks_with_caching
                new_messages_with_caching.append(new_message_copy)
        return super().__call__(
            messages=list(reversed(new_messages_with_caching)),
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )


class BaseAgent[T: BaseModel](AbstractAgent):
    def __init__(self, 
                    system_prompt: str, 
                    tools: list[Tool],
                    additional_authorized_imports: list[str],
                    final_answer_model: type[T] = BasicAnswerT,
                    final_answer_description: str = "The final answer to the user's question.",
                    model: str="anthropic/claude-sonnet-4-20250514",
                    max_steps: int=35,
                 ):
        self.system_prompt = system_prompt
        self.final_answer_model = final_answer_model
        self.final_answer_description = final_answer_description
        self.tools = tools
        self.tools.append(
            PydanticFinalAnswerTool(
                self.final_answer_model,
                description=self.final_answer_description
                or "The final answer to the user's question.",
            )
        )
        self.additional_authorized_imports = additional_authorized_imports
        self.max_steps = max_steps
        self.model = self._setup_model(model)
    
    def _setup_model(self, model: str) -> LiteLLMModel:
        if "anthropic" in model:
            return CachedAnthropicModel(model=model)
        else:
            return RetryingModel(model=model)
 
    def _setup_system_prompt(self, task: str) -> str:
        if "{task}" in self.system_prompt:
            return self.system_prompt.format(task=task)
        else:
            return self.system_prompt + "\n\n Task: " + task
    
    def format_step(self, step: ActionStep | PlanningStep | FinalAnswerStep | ChatMessageStreamDelta) -> StepT | ResultT[T]:
        if isinstance(step, FinalAnswerStep):
            # Convert final answer to ResultT
            return ResultT[T](answer=self.final_answer_model.model_validate(step.output))
        
        elif isinstance(step, ActionStep):
            parts = []
            step_number = step.step_number
            
            # Add thinking part if model output exists
            if step.model_output:
                content = step.model_output if isinstance(step.model_output, str) else str(step.model_output)
                parts.append(ThinkingPartT(
                    content=content
                ))
            
            # Add code part if code action exists
            if step.code_action:
                parts.append(CodePartT(
                    content=step.code_action
                ))
            
            # Add output part if observations exist
            if step.observations:
                parts.append(OutputPartT(
                    content=step.observations
                ))
            
            # If this is a final answer step, return the result instead
            if step.is_final_answer and step.action_output is not None:
                return ResultT[T](answer=self.final_answer_model.model_validate(step.action_output))
            
            return StepT(step_number=step_number, parts=parts)
        
        elif isinstance(step, PlanningStep):
            # Convert planning step to thinking part
            return StepT(step_number=None, parts=[ThinkingPartT(
                content=step.plan
            )])
        
        elif isinstance(step, ChatMessageStreamDelta):
            # Convert streaming content to thinking part
            return StepT(step_number=None, parts=[ThinkingPartT(
                content=step.content or ""
            )])
        
        else:
            # Fallback for unknown step types
            raise ValueError(f"Unknown step type: {type(step)}")
 
    def run(self, task: str, log: Callable[[StepT], None]) -> ResultT[T]:
        agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            additional_authorized_imports=self.additional_authorized_imports,
            max_steps=self.max_steps,
        )
        system_prompt = self._setup_system_prompt(task)
        final_result = None
        for step in agent.run(system_prompt, stream=True):
            formatted_step = self.format_step(step)
            if isinstance(formatted_step, ResultT):
                final_result = formatted_step
            else:
                log(formatted_step)
        if final_result is None:
            raise NoFinalResultError("No final result found")
        return final_result