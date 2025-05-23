# API Reference - Agentic Community Edition

## Overview

The Agentic Community Edition provides a simple yet powerful API for building AI agents. This reference covers all public classes and methods.

## Core Classes

### SimpleAgent

The main agent class for the community edition.

```python
class SimpleAgent(name: str, tools: List[BaseTool] = None)
```

#### Parameters
- `name` (str): The name of the agent
- `tools` (List[BaseTool], optional): List of tools to add to the agent (max 3)

#### Methods

##### `add_tool(tool: BaseTool) -> None`
Add a tool to the agent. Community edition supports up to 3 tools.

```python
agent.add_tool(SearchTool())
```

##### `run(task: str) -> str`
Execute a task and return the result.

```python
result = agent.run("Calculate 15% tip on $45.50")
```

##### `get_state() -> Dict[str, Any]`
Get the current state of the agent.

```python
state = agent.get_state()
```

##### `set_state(state: Dict[str, Any]) -> None`
Restore agent state from a dictionary.

```python
agent.set_state(saved_state)
```

##### `clear_history() -> None`
Clear the agent's conversation history.

```python
agent.clear_history()
```

## Tools

### BaseTool

Abstract base class for all tools.

```python
class BaseTool(ABC):
    name: str
    description: str
```

### SearchTool

Simulates web search functionality (mock in community edition).

```python
class SearchTool(BaseTool):
    name = "search"
    description = "Search for information on the web"
```

#### Usage
```python
search_tool = SearchTool()
agent.add_tool(search_tool)
```

### CalculatorTool

Performs mathematical calculations.

```python
class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Perform mathematical calculations"
```

#### Supported Operations
- Basic arithmetic: +, -, *, /, **, %
- Functions: sqrt, sin, cos, tan, log, exp
- Constants: pi, e

#### Usage
```python
calc_tool = CalculatorTool()
agent.add_tool(calc_tool)
```

### TextTool

Processes and manipulates text.

```python
class TextTool(BaseTool):
    name = "text_processor"
    description = "Process and manipulate text"
```

#### Capabilities
- Summarization
- Key point extraction
- Text formatting
- Word/character counting

#### Usage
```python
text_tool = TextTool()
agent.add_tool(text_tool)
```

## REST API

### Starting the Server

```bash
agentic serve [OPTIONS]
```

Options:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development

### Endpoints

#### `POST /agents/create`

Create a new agent instance.

Request:
```json
{
  "name": "Assistant",
  "tools": ["search", "calculator"]
}
```

Response:
```json
{
  "agent_id": "uuid-here",
  "name": "Assistant",
  "tools": ["search", "calculator"]
}
```

#### `POST /agents/{agent_id}/run`

Run a task with an agent.

Request:
```json
{
  "task": "What is 25% of 120?"
}
```

Response:
```json
{
  "result": "25% of 120 is 30",
  "agent_id": "uuid-here",
  "task_id": "task-uuid"
}
```

#### `GET /agents/{agent_id}/state`

Get agent state.

Response:
```json
{
  "agent_id": "uuid-here",
  "name": "Assistant",
  "tools": ["search", "calculator"],
  "history": []
}
```

#### `DELETE /agents/{agent_id}`

Delete an agent instance.

Response:
```json
{
  "message": "Agent deleted successfully"
}
```

#### `GET /health`

Health check endpoint.

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## CLI Commands

### `agentic serve`
Start the REST API server.

```bash
agentic serve --port 8080
```

### `agentic run`
Run a single task from the command line.

```bash
agentic run "Calculate 20% of 150" --tools calculator
```

### `agentic interactive`
Start an interactive session with an agent.

```bash
agentic interactive --name "Assistant" --tools search,calculator
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `AGENTIC_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `AGENTIC_MAX_RETRIES`: Maximum retries for failed operations (default: 3)

### Programmatic Configuration

```python
from agentic_community.core.utils import configure_logging

# Set logging level
configure_logging(level="DEBUG")
```

## Error Handling

### Common Exceptions

#### `ToolLimitExceeded`
Raised when trying to add more than 3 tools to an agent.

```python
try:
    agent.add_tool(fourth_tool)
except ToolLimitExceeded:
    print("Community edition supports max 3 tools")
```

#### `InvalidTaskError`
Raised when task input is invalid.

```python
try:
    agent.run("")  # Empty task
except InvalidTaskError:
    print("Task cannot be empty")
```

#### `APIKeyError`
Raised when OpenAI API key is missing or invalid.

```python
try:
    agent = SimpleAgent("Bot")
except APIKeyError:
    print("Please set OPENAI_API_KEY")
```

## Best Practices

### 1. Tool Selection
Choose tools that complement each other:
```python
# Good combination for research tasks
agent.add_tool(SearchTool())
agent.add_tool(TextTool())

# Good combination for calculations
agent.add_tool(CalculatorTool())
agent.add_tool(TextTool())
```

### 2. Error Handling
Always wrap agent operations in try-except blocks:
```python
try:
    result = agent.run(user_input)
except Exception as e:
    logger.error(f"Agent error: {e}")
    result = "I encountered an error processing your request."
```

### 3. State Management
Save agent state for persistence:
```python
# Save state
state = agent.get_state()
with open("agent_state.json", "w") as f:
    json.dump(state, f)

# Restore state
with open("agent_state.json", "r") as f:
    state = json.load(f)
agent.set_state(state)
```

### 4. Resource Management
Clear history for long-running agents:
```python
# Clear history every 50 interactions
if interaction_count % 50 == 0:
    agent.clear_history()
```

## Limitations

The Community Edition has the following limitations:
- Maximum 3 tools per agent
- Single agent execution only
- Basic sequential reasoning
- OpenAI LLM only
- No advanced features (self-reflection, multi-agent)

For advanced features, consider the [Enterprise Edition](https://agentic-ai.com/enterprise).

## Examples

See the [examples directory](../examples/) for complete working examples:
- `simple_example.py`: Basic usage
- `calculator_bot.py`: Math-focused agent
- `research_assistant.py`: Information gathering
- `task_planner.py`: Planning and organization

## Support

- GitHub Issues: [https://github.com/zahere/agentic-community/issues](https://github.com/zahere/agentic-community/issues)
- Discord: [https://discord.gg/agentic](https://discord.gg/agentic)
- Email: community@agentic-ai.com
