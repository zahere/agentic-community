# Getting Started with Agentic Community Edition

Welcome to the Agentic AI Framework Community Edition! This guide will help you get up and running with building your first AI agent.

## 📋 Prerequisites

Before you begin, ensure you have:
- Python 3.10 or higher installed
- An OpenAI API key (for LLM functionality)
- Basic familiarity with Python

## 🚀 Installation

### Using pip (Recommended)

```bash
pip install agentic-community
```

### From Source

```bash
git clone https://github.com/zahere/agentic-community.git
cd agentic-community
pip install -e .
```

## 🔑 Setting Up Your API Key

The community edition uses OpenAI for language model capabilities. Set your API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or in your Python code:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

## 🤖 Creating Your First Agent

Here's a simple example to get you started:

```python
from agentic_community import SimpleAgent, SearchTool, CalculatorTool

# Create an agent with a name
agent = SimpleAgent("MyAssistant")

# Add tools (community edition supports up to 3 tools)
agent.add_tool(SearchTool())
agent.add_tool(CalculatorTool())

# Run a task
result = agent.run("What is 25% of 840?")
print(result)
```

## 🛠️ Available Tools

The community edition includes three essential tools:

### 1. SearchTool
Simulates web search functionality (mock implementation in community edition).

```python
from agentic_community import SearchTool

search = SearchTool()
# Automatically used by agents for information queries
```

### 2. CalculatorTool
Performs mathematical calculations.

```python
from agentic_community import CalculatorTool

calculator = CalculatorTool()
# Handles math expressions, percentages, basic operations
```

### 3. TextTool
Processes and manipulates text.

```python
from agentic_community import TextTool

text_tool = TextTool()
# Summarization, extraction, formatting
```

## 💡 Example Use Cases

### Task Planning Assistant

```python
agent = SimpleAgent("TaskPlanner")
agent.add_tool(TextTool())

plan = agent.run("""
Help me plan a productive workday:
- 3 important meetings
- Need 2 hours for deep work
- Lunch break
- Email checking times
""")
print(plan)
```

### Research Assistant

```python
agent = SimpleAgent("Researcher")
agent.add_tool(SearchTool())
agent.add_tool(TextTool())

research = agent.run("""
Research the main benefits of meditation and 
summarize them in 3 bullet points
""")
print(research)
```

### Calculator Assistant

```python
agent = SimpleAgent("MathHelper")
agent.add_tool(CalculatorTool())

result = agent.run("""
Calculate my monthly budget:
- Income: $5,000
- Rent: $1,500
- Groceries: $400
- Utilities: $200
- What percentage is left for savings?
""")
print(result)
```

## 🌐 Using the REST API

Start the API server:

```bash
agentic serve --port 8000
```

Make requests:

```bash
curl -X POST http://localhost:8000/agents/run \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "Assistant",
    "task": "What is 2+2?",
    "tools": ["calculator"]
  }'
```

## 🎯 Best Practices

1. **Clear Task Descriptions**: Be specific about what you want the agent to do
2. **Tool Selection**: Only add tools that are relevant to your use case
3. **Error Handling**: Always handle potential errors in production code
4. **Rate Limiting**: Be mindful of API rate limits when using external services

## 🐛 Troubleshooting

### Import Errors
If you encounter import errors:
```bash
pip install --upgrade agentic-community
```

### API Key Issues
Ensure your OpenAI API key is valid and has sufficient credits.

### Memory Issues
For long-running tasks, consider breaking them into smaller subtasks.

## 📚 Next Steps

- Check out the [API Reference](api-reference.md) for detailed documentation
- Explore more [Examples](../examples/)
- Join our [Discord Community](https://discord.gg/agentic)
- Consider [Enterprise Edition](https://agentic-ai.com/enterprise) for advanced features

## 🤝 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discord**: For community support and discussions
- **Documentation**: For detailed guides and references

Happy building! 🚀
