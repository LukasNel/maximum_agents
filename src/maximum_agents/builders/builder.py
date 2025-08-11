import os
import tempfile
import shutil
from typing import Optional, Callable, List, Any, Dict
from pathlib import Path
from contextlib import contextmanager
import uuid
from PIL import Image
from io import BytesIO
from maximum_agents.records import ResultT
from openai import BaseModel
from smolagents import ActionStep, Tool

from ..datastore.core import MaximumDataStore
from ..datastore.types import SettingsT
from ..base import BaseAgent, HookRegistry
from ..document_types import DocumentT, DocumentsT

class DatabaseTool(Tool):
    name = "sqlengine"
    description = "Execute SQL queries on the database."
    inputs = {
        "query": {
            "type": "string",
            "description": "The SQL query to execute."
        }
    }
    output_type = "object"
    def __init__(self, database_id: str, datastore: MaximumDataStore):
        super().__init__()
        self.database_id = database_id
        self.datastore = datastore

    def forward(self, query: str):
        return self.datastore.sql_engine(self.database_id, query)


class AgentBuilder:
    """Builder class for creating agents with datastore and image processing capabilities using hook registry."""
    
    def __init__(self, hook_registry: Optional[HookRegistry] = None):
        self.hook_registry = hook_registry or HookRegistry()
        self._temp_dir: Optional[str] = None
        self._specific_dir: Optional[str] = None
        self._datastore: Optional[MaximumDataStore] = None
        self._database_id: Optional[str] = None
        self._image_adder: Optional[Callable[[], List[Image.Image]]] = None
        self._original_cwd: Optional[str] = None
        self.additional_tools: List[Tool] = []
        self.additional_imports: List[str] = []
    def put_agent_in_temporary_dir(self) -> 'AgentBuilder':
        """
        Configure the agent to run in a temporary directory.
        This prevents file creation in the main workspace.
        """
        self._temp_dir = tempfile.mkdtemp(prefix="agent_workspace_")
        
        # Add pre-run hook to change to temp directory
        def temp_dir_pre_run_hook(task: str) -> str:
            if self._temp_dir:
                self._original_cwd = os.getcwd()
                os.chdir(self._temp_dir)
            return task
        
        # Add post-run hook to change back directory (no document processing needed)
        def temp_dir_post_run_hook(task: str, result):
            # Change back to original directory
            if self._original_cwd:
                os.chdir(self._original_cwd)
            return result
        
        # Add error hook to ensure we change back even on errors
        def temp_dir_error_hook(error: Exception, task: str):
            if self._original_cwd:
                os.chdir(self._original_cwd)
            return None  # Don't handle the error, just cleanup
        
        # Add document_finder to final_answer_context
        def add_temp_document_finder_to_context(context: dict[str, Any]) -> dict[str, Any]:
            def document_finder(relative_path: str) -> str:
                """Convert relative path to absolute path based on temp directory."""
                if os.path.isabs(relative_path):
                    return relative_path
                else:
                    temp_dir = self._temp_dir or os.getcwd()
                    return os.path.abspath(os.path.join(temp_dir, relative_path))
            
            context['document_finder'] = document_finder
            return context
        
        self.hook_registry.add_pre_run_hook(temp_dir_pre_run_hook)
        self.hook_registry.add_post_run_hook(temp_dir_post_run_hook)
        self.hook_registry.add_error_hook(temp_dir_error_hook)
        self.hook_registry.add_final_answer_context_hook(add_temp_document_finder_to_context)
        
        return self
    
    def put_agent_in_specific_dir(self, directory_path: str) -> 'AgentBuilder':
        """
        Configure the agent to run in a specific directory.
        
        Args:
            directory_path: Path to the directory where the agent should run
        """
        # Ensure the directory exists
        os.makedirs(directory_path, exist_ok=True)
        self._specific_dir = os.path.abspath(directory_path)
        
        # Add pre-run hook to change to specific directory
        def specific_dir_pre_run_hook(task: str) -> str:
            if self._specific_dir:
                self._original_cwd = os.getcwd()
                os.chdir(self._specific_dir)
            return task
        
        # Add post-run hook to change back directory (no document processing needed)
        def specific_dir_post_run_hook(task: str, result):
            # Change back to original directory
            if self._original_cwd:
                os.chdir(self._original_cwd)
            return result
        
        # Add error hook to ensure we change back even on errors
        def specific_dir_error_hook(error: Exception, task: str):
            if self._original_cwd:
                os.chdir(self._original_cwd)
            return None  # Don't handle the error, just cleanup
        
        # Add document_finder to final_answer_context
        def add_document_finder_to_context(context: dict[str, Any]) -> dict[str, Any]:
            def document_finder(relative_path: str) -> str:
                """Convert relative path to absolute path based on specific directory."""
                if os.path.isabs(relative_path):
                    return relative_path
                else:
                    specific_dir = self._specific_dir or os.getcwd()
                    return os.path.abspath(os.path.join(specific_dir, relative_path))
            
            context['document_finder'] = document_finder
            return context
        
        self.hook_registry.add_pre_run_hook(specific_dir_pre_run_hook)
        self.hook_registry.add_post_run_hook(specific_dir_post_run_hook)
        self.hook_registry.add_error_hook(specific_dir_error_hook)
        self.hook_registry.add_final_answer_context_hook(add_document_finder_to_context)
        
        return self
    

    
    def add_database(self, datastore: MaximumDataStore, database_id: str) -> 'AgentBuilder':
        """
        Add a specific database from a MaximumDataStore to the agent and create SQL engine tool.
        
        Args:
            datastore: The MaximumDataStore instance to use
            database_id: The specific database ID to add
        """
        self._datastore = datastore
        self._database_id = database_id
        
        # Create SQL tool focused on the specific database
        sql_tool = DatabaseTool(database_id, datastore)
        self.additional_tools.append(sql_tool)
        
        # Add system prompt hook to include database description
        def database_system_prompt_hook(system_prompt: str, task: str) -> str:
            """Add database description to the system prompt."""
            if not self._datastore:
                return system_prompt
            
            try:
                # Check if the specific database exists
                if not self._datastore.backend.database_exists(database_id):
                    database_info = f"\n\nDatabase '{database_id}' does not exist or is not accessible."
                else:
                    # Get description for the specific database
                    description = self._datastore.get_database_description(database_id)
                    database_info = f"\n\nDatabase information:\n{description}"
                
                # Append database information to system prompt
                enhanced_prompt = system_prompt + database_info + "\n\nUse the sql_query tool to query this database when needed."
                return enhanced_prompt
                
            except Exception as e:
                # If there's an error getting database info, add error message
                database_info = f"\n\nDatabase '{database_id}': Error retrieving description - {str(e)}"
                enhanced_prompt = system_prompt + database_info
                return enhanced_prompt
        
        self.hook_registry.add_system_prompt_hook(database_system_prompt_hook)
        
        return self
    
    def add_imageadder(self, image_adder: Callable[[], List[Image.Image]]) -> 'AgentBuilder':
        """
        Add image processing capability to the agent.
        
        Args:
            image_adder: Function that returns a list of PIL Images to add to context
        """
        # Create step callback function for smolagents
        def process_visualizations(step_log : ActionStep) -> None:
            """Process and attach visualizations to the agent's step log."""
            print("process_visualizations")
            if not image_adder:
                return
                
            # Get images from the image_adder function
            images = image_adder()
            for image in images:
                image = image.convert('RGB')
            
            if images:
                # Add the images to the step log so the agent can see them
                step_log.observations_images = images
                
                # Update observations text to indicate images were added
                viz_text = f"Images: {len(images)} image(s) added to context"
                if step_log.observations is None:
                    step_log.observations = viz_text
                else:
                    step_log.observations = step_log.observations + "\n" + viz_text
        
        self.hook_registry.add_add_internal_step_hook(process_visualizations)
        
        return self
    
    def add_system_prompt_modifier(self, modifier: Callable[[str, str], str]) -> 'AgentBuilder':
        """
        Add a system prompt modifier hook.
        
        Args:
            modifier: Function that takes (system_prompt, task) and returns modified system_prompt
        """
        self.hook_registry.add_system_prompt_hook(modifier)
        return self
    
    def add_model_modifier(self, model_modifier: Callable[[str], str]) -> 'AgentBuilder':
        """
        Add a model setup modifier hook.
        
        Args:
            model_modifier: Function that takes model name and returns modified model name
        """
        self.hook_registry.add_model_setup_hook(model_modifier)
        return self
    
    def add_additional_tools(self, tools: List[Tool]) -> 'AgentBuilder':
        """
        Add additional tools to the agent.
        
        Args:
            tools: List of Tool objects to add
        """
        self.additional_tools.extend(tools)
        return self
    
    def add_additional_imports(self, imports: List[str]) -> 'AgentBuilder':
        """
        Add additional authorized imports to the agent.
        
        Args:
            imports: List of import names to add
        """
        self.additional_imports.extend(imports)
        return self


    def build_agent[T: BaseModel](self, *args, final_answer_model: type[T], final_answer_description: str, **kwargs):
        """
        Build and configure the agent with all added capabilities.
        
        Args:
            agent_class: The agent class to instantiate
            *args: Arguments to pass to agent constructor
            **kwargs: Keyword arguments to pass to agent constructor
            
        Returns:
            Configured agent instance
        """
        # Pass the hook registry to the agent
        kwargs['hook_registry'] = self.hook_registry
        kwargs['tools'] = kwargs.get('tools', []) + self.additional_tools
        kwargs['additional_authorized_imports'] = kwargs.get('additional_authorized_imports', []) + self.additional_imports
        # Create the agent
        agent = BaseAgent[T](*args, final_answer_model=final_answer_model, final_answer_description=final_answer_description, **kwargs)
        
        return agent
    
    def cleanup(self):
        """Clean up temporary directory and resources."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
        # Note: We don't clean up specific directories as they are user-provided
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()


# Convenience function for quick agent building
def create_agent_with_database(
    agent_class,
    datastore_settings: SettingsT,
    database_id: str,
    api_key: Optional[str] = None,
    use_temp_dir: bool = True,
    image_adder: Optional[Callable[[], List[Image.Image]]] = None,
    hook_registry: Optional[HookRegistry] = None,
    *args,
    **kwargs
):
    """
    Convenience function to quickly create an agent with database capabilities.
    
    Args:
        agent_class: The agent class to instantiate
        datastore_settings: Settings for the datastore
        database_id: The specific database ID to add
        api_key: Optional API key for the datastore
        use_temp_dir: Whether to use temporary directory
        image_adder: Optional image processing function
        hook_registry: Optional existing hook registry to use
        *args: Arguments to pass to agent constructor
        **kwargs: Keyword arguments to pass to agent constructor
        
    Returns:
        Configured agent instance
    """
    builder = AgentBuilder(hook_registry)
    
    if use_temp_dir:
        builder.put_agent_in_temporary_dir()
    
    # Create and add database
    datastore = MaximumDataStore(datastore_settings, api_key)
    builder.add_database(datastore, database_id)
    
    if image_adder:
        builder.add_imageadder(image_adder)
    
    return builder.build_agent(agent_class, *args, **kwargs)
