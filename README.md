# Agentic AI Framework - Community Edition

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/agentic-community.svg)](https://pypi.org/project/agentic-community/)
[![Discord](https://img.shields.io/discord/1234567890?color=7289da&label=Discord&logo=discord&logoColor=white)](https://discord.gg/agentic)

Build state-of-the-art autonomous AI agents with the Agentic AI Framework Community Edition - a powerful, open-source framework featuring MCP protocol support, vector databases, and enterprise-grade capabilities.

## 🚀 Features

### Community Edition Now Includes:
- 🤖 **Advanced Agent Architectures** - RAG-enhanced agents with semantic memory
- 🔧 **MCP Protocol Support** - Industry-standard tool interfaces (Anthropic MCP)
- 🧠 **Vector Database Integration** - Qdrant for semantic search and knowledge management
- 🛠️ **13+ Built-in Tools** - Search, calculator, file ops, web scraping, and more
- 🔌 **Plugin System** - Extend with custom tools and agents
- ⚡ **Real-time Communication** - WebSocket support for streaming responses
- 💾 **Intelligent Caching** - Performance optimization with flexible backends
- 🔒 **Enterprise Security** - JWT auth, rate limiting, API keys
- 📊 **Production Ready** - 85% test coverage, comprehensive docs

### 🆕 Latest Additions:
- **MCP (Model Context Protocol)** - Standardized tool interfaces
- **Qdrant Vector Search** - High-performance semantic similarity
- **RAG Implementation** - Retrieval-Augmented Generation
- **WebSocket Support** - Real-time agent communication
- **Plugin Architecture** - Easy extensibility

### Need More Power?
Check out our [Enterprise Edition](https://agentic-ai.com/enterprise) for:
- Multi-agent orchestration
- Advanced reasoning (Tree/Graph of Thoughts)
- All LLM providers (Anthropic, Google, Mistral)
- Distributed systems support
- Priority support and SLA
- Custom development services

## 📦 Installation

### Basic Installation
```bash
pip install agentic-community
```

### With Vector Database Support
```bash
pip install "agentic-community[vector]"
```

### With All Features
```bash
pip install "agentic-community[all]"
```

### From Source
```bash
git clone https://github.com/zahere/agentic-community.git
cd agentic-community
pip install -e ".[all]"
```

## 🏃 Quick Start

### Basic Agent
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

### RAG-Enhanced Agent
```python
from agentic_community.core.vector_store import QdrantVectorStore, RAGTool
from agentic_community import SimpleAgent

# Initialize vector store
vector_store = QdrantVectorStore()

# Create RAG tool
rag_tool = RAGTool(vector_store, embedding_function)

# Create agent with RAG
agent = SimpleAgent("Knowledge Assistant")
agent.add_tool(rag_tool)

# Add knowledge and query
agent.add_knowledge(["Important facts..."], source="docs")
result = agent.run("What are the important facts?")
```

## 📖 Documentation

- [Getting Started Guide](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [State-of-the-Art Features](docs/STATE_OF_THE_ART_ROADMAP.md)
- [Examples](examples/)
- [Contributing Guide](CONTRIBUTING.md)

## 🏗️ Architecture

The Agentic Framework follows a modular, extensible architecture:

```
agentic_community/
├── agents/          # Agent implementations
├── tools/           # 13+ built-in tools
├── core/            # Core components
│   ├── base/       # Base classes
│   ├── mcp/        # MCP protocol support
│   ├── vector_store.py  # Qdrant integration
│   ├── cache.py    # Caching layer
│   └── state/      # State management
├── plugins/         # Plugin system
├── api/            # REST + WebSocket APIs
└── examples/        # Example implementations
```

## 🎯 Use Cases

- **Knowledge Management** - RAG-powered document Q&A systems
- **Real-time Assistants** - WebSocket-based conversational agents
- **Research Tools** - Semantic search and information synthesis
- **Task Automation** - Complex workflow automation with tools
- **Custom Extensions** - Build domain-specific agents with plugins

## 🚀 Advanced Features

### MCP Protocol Support
```python
from agentic_community.core.mcp import MCPServer, create_mcp_server

# Create MCP server
server = create_mcp_server()

# Register tools with MCP
server.register_tool(SearchTool())

# Tools are now MCP-compliant!
```

### Vector Database & RAG
```python
# Semantic search with Qdrant
from agentic_community.core.vector_store import QdrantVectorStore

vector_store = QdrantVectorStore(
    collection_name="knowledge_base",
    host="localhost"
)

# Add documents with automatic chunking
knowledge_base.add_knowledge(documents, chunk_size=500)
```

### Plugin Development
```python
from agentic_community.plugins import create_plugin_template

# Generate plugin template
create_plugin_template("./my_plugin", "MyCustomTool")

# Load custom plugins
from agentic_community.plugins import load_plugin
plugin = load_plugin("./my_plugin/plugin.py")
```

## 📈 Roadmap

See our [State-of-the-Art Features Roadmap](docs/STATE_OF_THE_ART_ROADMAP.md) for upcoming features including:
- Multi-LLM provider support
- ReAct and Tree of Thoughts agents
- Advanced RAG techniques
- Observability with OpenTelemetry
- Edge deployment options

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

# Install in development mode with all features
pip install -e ".[all,dev]"

# Run tests
pytest

# Start Qdrant for vector tests
docker run -p 6333:6333 qdrant/qdrant
```

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
- [Qdrant](https://qdrant.tech/) - For vector search
- [MCP](https://github.com/anthropics/mcp) - For protocol standards
- [Pydantic](https://pydantic-docs.helpmanual.io/) - For data validation

## 📞 Support

- Community Edition: GitHub Issues & Discord
- Enterprise Edition: enterprise@agentic-ai.com

---

**Ready to build state-of-the-art AI agents? Get started today!**

```bash
pip install "agentic-community[all]"
```
