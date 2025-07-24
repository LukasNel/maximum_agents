from pydantic import BaseModel
import json

from smolagents.utils import extract_code_from_text
from .pydantic_final_answer_tools import PydanticFinalAnswerTool
from .abstract import AbstractAgent
from typing import Callable, Any, List, Dict, Optional, cast
from .records import  PartT, ResultT, BasicAnswerT, StepT, ThinkingPartT, CodePartT, OutputPartT, ToolCallT
from smolagents import CodeAgent, Tool, LiteLLMModel, ChatMessage, ChatMessageStreamDelta, ToolCall
from smolagents.agents import ToolOutput, ActionOutput
from smolagents.memory import ActionStep, PlanningStep, FinalAnswerStep
from typing import Generator
from litellm.exceptions import InternalServerError
from .exponential_backoff import exponential_backoff_agentonly

class NoFinalResultError(Exception):
    pass


# Hook type definitions
HookCallback = Callable[..., Any]
PreRunHook = Callable[[str], str]  # Takes task, returns potentially modified task
PostRunHook = Callable[[str, ResultT[Any]], ResultT[Any]]  # Takes task and result, returns potentially modified result
PreStepHook = Callable[[Any], Any]  # Takes step, returns potentially modified step
PostStepHook = Callable[[Any, StepT | ResultT[Any]], StepT | ResultT[Any]]  # Takes original step and formatted step, returns potentially modified formatted step
ErrorHook = Callable[[Exception, str], Optional[ResultT[Any]]]  # Takes exception and task, returns optional result to continue or None to re-raise
ModelSetupHook = Callable[[str], str]  # Takes model name, returns potentially modified model name
ModelSelectionHook = Callable[[str], LiteLLMModel]  # Takes model name, returns instantiated model
CodeAgentKwargsHook = Callable[[], Dict[str, Any]]  # Returns additional kwargs for CodeAgent constructor
SystemPromptHook = Callable[[str, str], str]  # Takes system_prompt and task, returns potentially modified system_prompt


class HookRegistry:
    """Registry for managing hooks of different types."""
    
    def __init__(self):
        self.pre_run_hooks: List[PreRunHook] = []
        self.post_run_hooks: List[PostRunHook] = []
        self.pre_step_hooks: List[PreStepHook] = []
        self.post_step_hooks: List[PostStepHook] = []
        self.error_hooks: List[ErrorHook] = []
        self.model_setup_hooks: List[ModelSetupHook] = []
        self.model_selection_hooks: List[ModelSelectionHook] = []
        self.codeagent_kwargs_hooks: List[CodeAgentKwargsHook] = []
        self.system_prompt_hooks: List[SystemPromptHook] = []
    
    def add_pre_run_hook(self, hook: PreRunHook):
        """Add a hook that runs before agent execution starts."""
        self.pre_run_hooks.append(hook)
    
    def add_post_run_hook(self, hook: PostRunHook):
        """Add a hook that runs after agent execution completes."""
        self.post_run_hooks.append(hook)
    
    def add_pre_step_hook(self, hook: PreStepHook):
        """Add a hook that runs before each step is processed."""
        self.pre_step_hooks.append(hook)
    
    def add_post_step_hook(self, hook: PostStepHook):
        """Add a hook that runs after each step is formatted."""
        self.post_step_hooks.append(hook)
    
    def add_error_hook(self, hook: ErrorHook):
        """Add a hook that runs when an error occurs during execution."""
        self.error_hooks.append(hook)
    
    def add_model_setup_hook(self, hook: ModelSetupHook):
        """Add a hook that runs during model setup."""
        self.model_setup_hooks.append(hook)
    
    def add_model_selection_hook(self, hook: ModelSelectionHook):
        """Add a hook that runs during model selection and instantiation."""
        self.model_selection_hooks.append(hook)
    
    def add_codeagent_kwargs_hook(self, hook: CodeAgentKwargsHook):
        """Add a hook that provides additional kwargs for CodeAgent constructor."""
        self.codeagent_kwargs_hooks.append(hook)
    
    def add_system_prompt_hook(self, hook: SystemPromptHook):
        """Add a hook that runs during system prompt setup."""
        self.system_prompt_hooks.append(hook)
    
    def clear_hooks(self, hook_type: Optional[str] = None):
        """Clear hooks of a specific type or all hooks if hook_type is None."""
        if hook_type is None:
            self.pre_run_hooks.clear()
            self.post_run_hooks.clear()
            self.pre_step_hooks.clear()
            self.post_step_hooks.clear()
            self.error_hooks.clear()
            self.model_setup_hooks.clear()
            self.model_selection_hooks.clear()
            self.codeagent_kwargs_hooks.clear()
            self.system_prompt_hooks.clear()
        elif hook_type == "pre_run":
            self.pre_run_hooks.clear()
        elif hook_type == "post_run":
            self.post_run_hooks.clear()
        elif hook_type == "pre_step":
            self.pre_step_hooks.clear()
        elif hook_type == "post_step":
            self.post_step_hooks.clear()
        elif hook_type == "error":
            self.error_hooks.clear()
        elif hook_type == "model_setup":
            self.model_setup_hooks.clear()
        elif hook_type == "model_selection":
            self.model_selection_hooks.clear()
        elif hook_type == "codeagent_kwargs":
            self.codeagent_kwargs_hooks.clear()
        elif hook_type == "system_prompt":
            self.system_prompt_hooks.clear()
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")


def default_model_selection_hook(model: str) -> LiteLLMModel:
    """Default model selection hook that chooses between CachedAnthropicModel and RetryingModel."""
    if "anthropic" in model:
        return CachedAnthropicModel(model_id=model)
    else:
        return RetryingModel(model_id=model)


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

def clear_code_from_text_and_return_seperate_text(text: str, code_block_tags: tuple[str, str]) -> tuple[str, str | None]:
    code_action = extract_code_from_text(text, code_block_tags)
    if code_action:
        text = text.replace(code_action, "")
    return text, code_action

def content_to_thinking_and_optionally_code(content: str, code_block_tags: tuple[str, str]) -> list[PartT]:
    text, code_action = clear_code_from_text_and_return_seperate_text(content, code_block_tags)
    parts = []
    if code_action:
        parts.append(CodePartT(content=code_action))
    if text:
        parts.append(ThinkingPartT(content=text))
    return parts

class BaseAgent[T: BaseModel](AbstractAgent):
    def __init__(self, 
                    system_prompt: str, 
                    tools: list[Tool],
                    additional_authorized_imports: list[str],
                    final_answer_model: type[T] = BasicAnswerT,
                    final_answer_description: str = "The final answer to the user's question.",
                    model: str="anthropic/claude-sonnet-4-20250514",
                    max_steps: int=35,
                    hook_registry: Optional[HookRegistry] = None,
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
        self.hooks = hook_registry or HookRegistry()  # Use provided registry or create new one
        self.agent : CodeAgent | None = None
        # Set up default model selection hook if none exists
        if not self.hooks.model_selection_hooks:
            self.hooks.add_model_selection_hook(default_model_selection_hook)
        self.hooks.add_system_prompt_hook(self._add_task_to_system_prompt)
        self.hooks.add_system_prompt_hook(self._add_final_answer_description_to_system_prompt)
        self.model = self._setup_model(model)

    def _add_task_to_system_prompt(self, system_prompt: str, task: str) -> str:
        system_prompt = system_prompt + "\n\n Task: " + task
        return system_prompt
    
    def _add_final_answer_description_to_system_prompt(self, system_prompt: str, task: str) -> str:
        system_prompt = system_prompt + "\n\n" + self.final_answer_description + "\n\n" + json.dumps(self.final_answer_model.model_json_schema())
        return system_prompt
    
    def _setup_model(self, model: str) -> LiteLLMModel:
        # Apply model setup hooks to modify the model name
        for hook in self.hooks.model_setup_hooks:
            model = hook(model)
        
        # Use model selection hooks to instantiate the model
        # If multiple hooks are registered, the last one takes precedence
        if self.hooks.model_selection_hooks:
            return self.hooks.model_selection_hooks[-1](model)
        else:
            # Fallback to default behavior if no hooks are registered
            return default_model_selection_hook(model)
 
    def _setup_system_prompt(self, task: str) -> str:
        # Apply system prompt hooks
        system_prompt = self.system_prompt
        for hook in self.hooks.system_prompt_hooks:
            system_prompt = hook(system_prompt, task)
        
        return system_prompt
    
    def format_step(self,step_number : int, step: ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput | ActionStep | PlanningStep | FinalAnswerStep) -> StepT | ResultT[T]:
        assert self.agent is not None
        if isinstance(step, ChatMessageStreamDelta):
            # Convert streaming content to thinking part
            return StepT(step_number=step_number, parts=[ThinkingPartT(
                content=step.content or ""
            )])
        
        elif isinstance(step, ToolOutput):
            # Convert tool output to output part
            observation = step.observation
            code_action = extract_code_from_text(observation, self.agent.code_block_tags)
            parts = []

            if code_action:
                observation = observation.replace(code_action, "")
                parts.append(CodePartT(content=code_action))
            if observation:
                parts.append(OutputPartT(content=observation))
            return StepT(step_number=step_number, parts=parts)
        
        elif isinstance(step, ActionOutput):
            # Convert action output - if final answer, return ResultT, otherwise OutputPartT
            if step.is_final_answer:
                return ResultT[T](answer=self.final_answer_model.model_validate(step.output))
            else:
                return StepT(step_number=step_number, parts=[OutputPartT(
                    content=str(step.output)
                )])
        
        elif isinstance(step, ToolCall):
            # Handle tool calls - python_interpreter becomes CodePartT, others become ToolCallT
            tool_dict = cast(Dict[str, Any], step.dict())
            function_info = cast(Dict[str, Any], tool_dict.get("function", {}))
            tool_name = str(function_info.get("name", ""))
            tool_arguments = function_info.get("arguments", {})
            
            if tool_name == "python_interpreter":
                return StepT(step_number=step_number, parts=[CodePartT(content=str(tool_arguments))])
            else:
                # Format other tool calls as ToolCallT
                # Ensure arguments is a dict
                if not isinstance(tool_arguments, dict):
                    tool_arguments = {"value": tool_arguments}
                return StepT(step_number=step_number, parts=[ToolCallT(name=tool_name, arguments=cast(Dict[str, Any], tool_arguments))])
        
        elif isinstance(step, ActionStep):
            # Handle ActionStep - extract different parts and separate code blocks
            parts = []
            
            # If this is a final answer, return ResultT
            if step.is_final_answer and step.action_output is not None:
                return ResultT[T](answer=self.final_answer_model.model_validate(step.action_output))
            
            # Handle model output (thinking/reasoning text with potential code blocks)
            if step.model_output:
                if isinstance(step.model_output, str):
                    model_parts = content_to_thinking_and_optionally_code(step.model_output, self.agent.code_block_tags)
                    parts.extend(model_parts)
                else:
                    # Handle list format - convert to string first
                    model_output_str = str(step.model_output)
                    model_parts = content_to_thinking_and_optionally_code(model_output_str, self.agent.code_block_tags)
                    parts.extend(model_parts)
            
            # Handle separate code action if present
            if step.code_action:
                parts.append(CodePartT(content=step.code_action))
            
            # Handle observations (tool outputs, execution results)
            if step.observations:
                observation_parts = content_to_thinking_and_optionally_code(step.observations, self.agent.code_block_tags)
                # Convert thinking parts from observations to output parts
                for part in observation_parts:
                    if isinstance(part, ThinkingPartT):
                        parts.append(OutputPartT(content=part.content))
                    else:
                        parts.append(part)
            
            # Handle action output (if not final answer)
            if step.action_output is not None and not step.is_final_answer:
                action_output_str = str(step.action_output)
                output_parts = content_to_thinking_and_optionally_code(action_output_str, self.agent.code_block_tags)
                # Convert thinking parts to output parts for action outputs
                for part in output_parts:
                    if isinstance(part, ThinkingPartT):
                        parts.append(OutputPartT(content=part.content))
                    else:
                        parts.append(part)
            
            return StepT(step_number=step_number, parts=parts)
        
        elif isinstance(step, PlanningStep):
            # Handle PlanningStep - extract plan text and separate code blocks
            parts = []
            
            if step.plan:
                plan_parts = content_to_thinking_and_optionally_code(step.plan, self.agent.code_block_tags)
                parts.extend(plan_parts)
            
            return StepT(step_number=step_number, parts=parts)
        
        elif isinstance(step, FinalAnswerStep):
            # Handle FinalAnswerStep - this should be the final result
            if step.output is not None:
                return ResultT[T](answer=self.final_answer_model.model_validate(step.output))
            else:
                # If no output, treat as empty step
                return StepT(step_number=step_number, parts=[])
        
        else:
            # Fallback for unknown step types
            raise ValueError(f"Unknown step type: {type(step)}")

    def _execute_pre_run_hooks(self, task: str) -> str:
        """Execute all pre-run hooks in sequence."""
        modified_task = task
        for hook in self.hooks.pre_run_hooks:
            modified_task = hook(modified_task)
        return modified_task
    
    def _execute_post_run_hooks(self, task: str, result: ResultT[T]) -> ResultT[T]:
        """Execute all post-run hooks in sequence."""
        modified_result = result
        for hook in self.hooks.post_run_hooks:
            modified_result = hook(task, modified_result)
        return modified_result
    
    def _execute_pre_step_hooks(self, step: Any) -> Any:
        """Execute all pre-step hooks in sequence."""
        modified_step = step
        for hook in self.hooks.pre_step_hooks:
            modified_step = hook(modified_step)
        return modified_step
    
    def _execute_post_step_hooks(self, original_step: Any, formatted_step: StepT | ResultT[T]) -> StepT | ResultT[T]:
        """Execute all post-step hooks in sequence."""
        modified_formatted_step = formatted_step
        for hook in self.hooks.post_step_hooks:
            modified_formatted_step = hook(original_step, modified_formatted_step)
        return modified_formatted_step
    
    def _execute_error_hooks(self, error: Exception, task: str) -> Optional[ResultT[T]]:
        """Execute all error hooks until one returns a result or all return None."""
        for hook in self.hooks.error_hooks:
            result = hook(error, task)
            if result is not None:
                return result
        return None
    
    def _execute_codeagent_kwargs_hooks(self) -> Dict[str, Any]:
        """Execute all CodeAgent kwargs hooks and merge their results."""
        additional_kwargs = {}
        for hook in self.hooks.codeagent_kwargs_hooks:
            hook_kwargs = hook()
            if hook_kwargs:
                additional_kwargs.update(hook_kwargs)
        return additional_kwargs

    def run(self, task: str, log: Callable[[StepT], None]) -> ResultT[T]:
        try:
            # Execute pre-run hooks
            task = self._execute_pre_run_hooks(task)
            
            # Collect additional kwargs from hooks
            additional_kwargs = self._execute_codeagent_kwargs_hooks()
            
            # Create CodeAgent with base parameters and additional kwargs
            print(self.tools)
            self.agent = CodeAgent(
                tools=self.tools,
                model=self.model,
                additional_authorized_imports=self.additional_authorized_imports,
                max_steps=self.max_steps,
                **additional_kwargs
            )
            system_prompt = self._setup_system_prompt(task)
            final_result = None
            
            # Use streaming approach - returns a generator that yields steps
            step_generator =  self.agent.run(system_prompt, stream=True)
            step_number = 1
            for step in step_generator:
                # Execute pre-step hooks
                step = self._execute_pre_step_hooks(step)
                
                # Format the step
                formatted_step = self.format_step(step_number, step)
                
                # Execute post-step hooks
                formatted_step = self._execute_post_step_hooks(step, formatted_step)
                
                if isinstance(formatted_step, ResultT):
                    final_result = formatted_step
                    break  # We found our final result
                else:
                    log(formatted_step)
                step_number += 1
            
            if final_result is None:
                raise NoFinalResultError("No final result found")
            
            # Execute post-run hooks
            final_result = self._execute_post_run_hooks(task, final_result)
            
            return final_result
            
        except Exception as e:
            # Try to handle the error with error hooks
            recovery_result = self._execute_error_hooks(e, task)
            if recovery_result is not None:
                return recovery_result
            # If no hook handled it, re-raise the original exception
            raise