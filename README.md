# Agentic AI Framework - Community Edition

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/agentic-community.svg)](https://pypi.org/project/agentic-community/)
[![Discord](https://img.shields.io/discord/1234567890?color=7289da&label=Discord&logo=discord&logoColor=white)](https://discord.gg/agentic)

Build autonomous AI agents with the Agentic AI Framework Community Edition - a powerful, open-source framework for creating intelligent agents that can reason, plan, and execute tasks.

## 🚀 Features

### Community Edition Includes:
- 🤖 **Basic Sequential Reasoning** - Simple yet effective task decomposition
- 🔧 **Task Planning** - Break complex tasks into manageable steps  
- 🛠️ **Essential Tools** - Search, calculator, and text processing
- 🏃 **Single Agent Execution** - Run one agent at a time
- 📦 **Easy Deployment** - Simple setup and configuration
- 🌐 **REST API** - Build web services with your agents
- 📚 **Examples & Tutorials** - Get started quickly

### Need More Power?
Check out our [Enterprise Edition](https://agentic-ai.com/enterprise) for:
- Advanced reasoning with self-reflection
- Multi-agent orchestration
- Game theory integration
- Support for all LLM providers
- Priority support and SLA
- Custom development services

## 📦 Installation

```bash
pip install agentic-community
```

Or install from source:

```bash
git clone https://github.com/zahere/agentic-community.git
cd agentic-community
pip install -e .
```

## 🏃 Quick Start

```python
import os
from agentic_community import SimpleAgent, SearchTool, CalculatorTool

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Create an agent
agent = SimpleAgent("Assistant")

# Add tools (community edition supports up to 3 tools)
agent.add_tool(SearchTool())
agent.add_tool(CalculatorTool())

# Run a task
result = agent.run("Help me plan a weekend trip to Paris with a $2000 budget")
print(result)
```

## 📖 Documentation

- [Getting Started Guide](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Examples](examples/)
- [Contributing Guide](CONTRIBUTING.md)

## 🏗️ Architecture

The Agentic Framework follows a modular architecture:

```
agentic_community/
├── agents/          # Agent implementations
├── tools/           # Available tools
├── core/            # Core components
│   ├── base/       # Base classes
│   ├── reasoning/  # Reasoning engine
│   └── state/      # State management
└── examples/        # Example implementations
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repo
git clone https://github.com/zahere/agentic-community.git
cd agentic-community

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

## 🎯 Use Cases

- **Personal Assistants** - Build AI assistants for daily tasks
- **Content Creation** - Generate and process content
- **Data Analysis** - Analyze and summarize information
- **Task Automation** - Automate repetitive workflows
- **Educational Tools** - Create learning assistants

## 🚧 Limitations

The Community Edition has some limitations:
- Single agent execution only
- Maximum of 3 tools per agent
- Basic sequential reasoning (no self-reflection)
- OpenAI LLM support only
- Community support only

For advanced features, consider upgrading to [Enterprise Edition](https://agentic-ai.com/enterprise).

## 📈 Roadmap

- [ ] More tool integrations
- [ ] Improved reasoning capabilities
- [ ] Better error handling
- [ ] Performance optimizations
- [ ] Additional examples
- [ ] Video tutorials

## 💬 Community

- [Discord Server](https://discord.gg/agentic)
- [GitHub Discussions](https://github.com/zahere/agentic-community/discussions)
- [Twitter](https://twitter.com/agenticai)

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- [LangChain](https://langchain.com/) - For LLM orchestration
- [LangGraph](https://github.com/langchain-ai/langgraph) - For agent workflows
- [Pydantic](https://pydantic-docs.helpmanual.io/) - For data validation

## 📞 Support

- Community Edition: GitHub Issues & Discord
- Enterprise Edition: enterprise@agentic-ai.com

---

**Ready to build amazing AI agents? Get started today!**

```bash
pip install agentic-community
```
