# Maximum Agents Framework

A powerful Python framework for building intelligent agents with structured outputs, document generation, database integration, and advanced hook-based customization.

## 🚀 Features

- **Structured Output Models**: Define complex Pydantic models for agent responses
- **Document Generation**: Automatic file creation with path tracking and validation
- **Database Integration**: Built-in SQL engine with automatic schema descriptions
- **Hook System**: Comprehensive hook registry for customizing agent behavior
- **Image Processing**: Support for visualization and image analysis
- **Workspace Management**: Temporary or specific directory execution
- **Model Flexibility**: Support for multiple LLM providers with retry logic
- **Step Monitoring**: Real-time progress tracking and logging

## 📦 Installation

```bash
pip install maximum-agents
```

## 🎯 Quick Start

### Basic Agent

```python
from maximum_agents import BaseAgent
from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
    summary: str = Field(description="Analysis summary")
    confidence: float = Field(description="Confidence score (0-1)")

# Create a simple agent
agent = BaseAgent(
    system_prompt="You are a data analyst.",
    tools=[],
    additional_authorized_imports=["pandas", "numpy"],
    final_answer_model=AnalysisResult,
    final_answer_description="The analysis results"
)

# Run the agent
def log_step(step):
    print(f"Step {step.step_number}: {step.parts[0].content[:50]}...")

result = agent.run("Analyze this dataset: [1,2,3,4,5]", log_step)
print(f"Summary: {result.answer.summary}")
print(f"Confidence: {result.answer.confidence}")
```

### Advanced Agent with Document Generation

```python
from maximum_agents import AgentBuilder, DocumentT, DocumentsT
from maximum_agents.tools import WebSearchTool, GetDocumentTool
from pydantic import BaseModel, Field
from typing import List

class ResearchOutput(BaseModel):
    executive_summary: str = Field(description="Key findings summary")
    methodology: str = Field(description="Research approach used")
    main_document: DocumentT = Field(description="Primary research report")
    supporting_charts: List[DocumentT] = Field(description="Generated visualizations")
    confidence_score: float = Field(description="Research confidence (0-1)")

# Build agent with workspace and tools
builder = AgentBuilder()
builder.put_agent_in_specific_dir("/tmp/research_workspace")
builder.add_additional_tools([WebSearchTool(), GetDocumentTool()])

# Create agent with complex structured output
agent = builder.build_agent(
    system_prompt="You are a research analyst who creates comprehensive reports with visualizations.",
    additional_authorized_imports=["matplotlib", "pandas", "python-docx"],
    final_answer_model=ResearchOutput,
    final_answer_description="Comprehensive research analysis with documents and charts",
    max_steps=25
)

# Run with step monitoring
def monitor_progress(step):
    for part in step.parts:
        if part.type == "THINKING":
            print(f"Planning: {part.content[:80]}...")
        elif part.type == "CODE":
            print(f"Executing: {part.content.split()[0]}...")
        elif part.type == "OUTPUT":
            print(f"Result: {part.content[:60]}...")

result = agent.run(
    "Analyze the electric vehicle market trends for 2025 and create a comprehensive report with charts", 
    monitor_progress
)

# Access structured results with automatic path resolution
research_output = result.answer
print(f"Summary: {research_output.executive_summary}")
print(f"Main report: {research_output.main_document.absolute_path}")
print(f"Charts generated: {len(research_output.supporting_charts)}")
```

## 🏗️ Architecture

### Core Components

- **BaseAgent**: Main agent class with hook system and structured outputs
- **AgentBuilder**: Fluent API for configuring agents with capabilities
- **HookRegistry**: Comprehensive hook system for customization
- **DocumentT/DocumentsT**: Document types with automatic path resolution
- **RetryingModel**: Robust model wrapper with exponential backoff

### Hook System

The framework provides extensive hooks for customization:

```python
from maximum_agents import HookRegistry, BaseAgent

# Create custom hook registry
hooks = HookRegistry()

# Add pre-run hook
def pre_run_hook(task: str) -> str:
    return f"Enhanced task: {task}"

hooks.add_pre_run_hook(pre_run_hook)

# Add model selection hook
def custom_model_hook(model_name: str) -> LiteLLMModel:
    return RetryingModel(model_id=model_name, temperature=0.7)

hooks.add_model_selection_hook(custom_model_hook)

# Use with agent
agent = BaseAgent(
    system_prompt="You are an AI assistant.",
    tools=[],
    additional_authorized_imports=[],
    final_answer_model=BasicAnswerT,
    hook_registry=hooks
)
```

## 📊 Document Management

### Document Types

```python
from maximum_agents import DocumentT, DocumentsT

class ReportOutput(BaseModel):
    title: str
    main_report: DocumentT = Field(description="Primary report document")
    appendices: List[DocumentT] = Field(description="Supporting documents")
    charts: DocumentsT = Field(description="Generated charts and visualizations")
```

### Automatic Path Resolution

Documents automatically resolve absolute paths:

```python
# Agent creates file: "reports/analysis.pdf"
document = DocumentT(
    path="reports/analysis.pdf",
    explanation="Market analysis report"
)

# Automatically resolves to absolute path
print(document.absolute_path)  # /tmp/research_workspace/reports/analysis.pdf
```

## 🗄️ Database Integration

### SQL Engine Tool

```python
from maximum_agents import AgentBuilder
from maximum_agents.datastore import MaximumDataStore, SettingsT

# Configure datastore
settings = SettingsT(
    backend_type="duckdb",
    connection_string=":memory:"
)

datastore = MaximumDataStore(settings)
datastore.create_database("sales_data")

# Add database to agent
builder = AgentBuilder()
builder.add_database(datastore, "sales_data")

agent = builder.build_agent(
    system_prompt="You are a data analyst with SQL access.",
    tools=[],
    additional_authorized_imports=["pandas"],
    final_answer_model=AnalysisResult
)

# Agent can now use SQL queries via the sqlengine tool
result = agent.run("Query the sales data and find top products", log_step)
```

## 🖼️ Image Processing

### Visualization Support

```python
from PIL import Image
from maximum_agents import AgentBuilder

def generate_charts() -> List[Image.Image]:
    """Generate visualization charts"""
    # Your chart generation logic
    return [chart1, chart2]

# Add image processing to agent
builder = AgentBuilder()
builder.add_imageadder(generate_charts)

agent = builder.build_agent(
    system_prompt="You are a data visualization expert.",
    tools=[],
    additional_authorized_imports=["matplotlib", "seaborn"],
    final_answer_model=VisualizationResult
)
```

## ⚙️ Configuration

### Model Configuration

```python
from maximum_agents import BaseAgent, RetryingModel, CachedAnthropicModel

# Use specific model with custom parameters
agent = BaseAgent(
    system_prompt="You are an AI assistant.",
    tools=[],
    additional_authorized_imports=[],
    final_answer_model=BasicAnswerT,
    model="anthropic/claude-sonnet-4-20250514",
    model_kwargs={"temperature": 0.7, "max_tokens": 2000}
)

# Or use custom model instance
custom_model = RetryingModel(
    model_id="gpt-4",
    temperature=0.5,
    max_retries=3
)

agent = BaseAgent(
    system_prompt="You are an AI assistant.",
    tools=[],
    additional_authorized_imports=[],
    final_answer_model=BasicAnswerT,
    model=custom_model
)
```

### Workspace Management

```python
from maximum_agents import AgentBuilder

builder = AgentBuilder()

# Use temporary directory (default)
builder.put_agent_in_temporary_dir()

# Or use specific directory
builder.put_agent_in_specific_dir("/path/to/workspace")

# Add additional tools and imports
builder.add_additional_tools([CustomTool()])
builder.add_additional_imports(["requests", "beautifulsoup4"])

agent = builder.build_agent(
    system_prompt="You are a web scraper.",
    tools=[],
    additional_authorized_imports=[],
    final_answer_model=ScrapingResult
)
```

## 🔧 Advanced Usage

### Custom Tools

```python
from smolagents import Tool
from maximum_agents import BaseAgent

class CustomAnalysisTool(Tool):
    name = "custom_analysis"
    description = "Perform custom data analysis"
    inputs = {
        "data": {"type": "string", "description": "Data to analyze"}
    }
    output_type = "object"
    
    def forward(self, data: str):
        # Your analysis logic
        return {"result": "analysis_complete"}

# Use custom tool
agent = BaseAgent(
    system_prompt="You are a data analyst.",
    tools=[CustomAnalysisTool()],
    additional_authorized_imports=["pandas"],
    final_answer_model=AnalysisResult
)
```

### Error Handling

```python
from maximum_agents import HookRegistry

def error_recovery_hook(error: Exception, task: str):
    """Handle errors gracefully"""
    if "rate limit" in str(error).lower():
        return ResultT(answer=BasicAnswerT(answer="Rate limited, please try again later"))
    return None  # Re-raise other errors

hooks = HookRegistry()
hooks.add_error_hook(error_recovery_hook)

agent = BaseAgent(
    system_prompt="You are an AI assistant.",
    tools=[],
    additional_authorized_imports=[],
    final_answer_model=BasicAnswerT,
    hook_registry=hooks
)
```

## 📚 API Reference

### BaseAgent

```python
class BaseAgent[T: BaseModel]:
    def __init__(
        self,
        system_prompt: str,
        tools: list[Tool],
        additional_authorized_imports: list[str],
        max_print_outputs_length: int = 50000,
        final_answer_model: type[T] = BasicAnswerT,
        final_answer_description: str = "The final answer to the user's question.",
        model: Union[str, LiteLLMModel] = "anthropic/claude-sonnet-4-20250514",
        model_kwargs: dict[str, Any] = {},
        max_steps: int = 35,
        hook_registry: Optional[HookRegistry] = None,
        final_answer_context: dict[str, Any] = {},
    )
    
    def run(self, task: str, log: Callable[[StepT], None]) -> ResultT[T]
```

### AgentBuilder

```python
class AgentBuilder:
    def put_agent_in_temporary_dir(self) -> 'AgentBuilder'
    def put_agent_in_specific_dir(self, directory_path: str) -> 'AgentBuilder'
    def add_database(self, datastore: MaximumDataStore, database_id: str) -> 'AgentBuilder'
    def add_imageadder(self, image_adder: Callable[[], List[Image.Image]]) -> 'AgentBuilder'
    def add_additional_tools(self, tools: List[Tool]) -> 'AgentBuilder'
    def add_additional_imports(self, imports: List[str]) -> 'AgentBuilder'
    def build_agent[T: BaseModel](self, *args, final_answer_model: type[T], **kwargs) -> BaseAgent[T]
```

### Document Types

```python
class DocumentT(BaseModel):
    path: str = Field(description="File path of the generated document")
    explanation: str = Field(description="Explanation of what the document contains")
    absolute_path: Optional[str] = Field(default=None, description="Absolute file path")

class DocumentsT(BaseModel):
    documents: List[DocumentT] = Field(description="List of documents generated")
```

## 🤝 Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/LukasNel/maximum_agents) for:

- Issue reporting
- Feature requests
- Pull requests
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [GitHub Repository](https://github.com/LukasNel/maximum_agents)
- [PyPI Package](https://pypi.org/project/maximum-agents/)
- [Documentation](https://maximum-agents.readthedocs.io/)
- [Publishing Guide](PUBLISHING.md)

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/LukasNel/maximum_agents/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LukasNel/maximum_agents/discussions)
- **Email**: lukas@lotushealth.ai

---

Built with ❤️ by the Maximum Agents team
