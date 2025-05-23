"""
Agentic AI Framework - Community Edition
Setup configuration for pip installation
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="agentic-community",
    version="1.0.0",
    author="Zaher Khateeb",
    author_email="zaher.master@gmail.com",
    description="Agentic AI Framework Community Edition - Build autonomous AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zahere/agentic-community",
    project_urls={
        "Bug Reports": "https://github.com/zahere/agentic-community/issues",
        "Source": "https://github.com/zahere/agentic-community",
        "Documentation": "https://docs.agentic-ai.com",
        "Discord": "https://discord.gg/agentic",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    keywords="ai, agents, autonomous, reasoning, lightweight, mcp, rag, vector-search",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    python_requires=">=3.10, <4",
    install_requires=[
        # Core dependencies
        "typing-extensions>=4.9.0",
        "pydantic>=2.0",
        # LangChain dependencies (required for agents)
        "langchain-core>=0.2.0",
        "langchain-community>=0.1.0",
        "langchain-openai>=0.1.0",
        "langgraph>=0.4.0",
        # API dependencies for REST functionality
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "aiohttp>=3.9.0",
        # WebSocket support
        "websockets>=12.0",
        # Search and web scraping
        "beautifulsoup4>=4.12.0",
        "duckduckgo-search>=6.1.0",
        # Data processing
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        # Authentication
        "PyJWT>=2.8.0",
        "passlib>=1.7.4",
        "python-multipart>=0.0.9",
        # Performance monitoring
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.13.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
            "myst-parser>=2.0.0",
        ],
        "vector": [
            # Vector database support
            "qdrant-client>=1.7.0",
            # Embedding models
            "openai>=1.0.0",  # For OpenAI embeddings
            "cohere>=4.0.0",  # For Cohere embeddings
            "sentence-transformers>=2.2.0",  # For local embeddings
        ],
        "mcp": [
            # MCP protocol support
            "jsonschema>=4.20.0",
            "httpx>=0.25.0",
        ],
        "advanced": [
            # Advanced agent architectures
            "networkx>=3.0",  # For graph-based reasoning
            "scikit-learn>=1.3.0",  # For ML utilities
            # Observability
            "opentelemetry-api>=1.20.0",
            "opentelemetry-sdk>=1.20.0",
            "opentelemetry-instrumentation-fastapi>=0.41b0",
            # Additional LLM providers
            "anthropic>=0.7.0",
            "google-generativeai>=0.3.0",
            "mistralai>=0.0.8",
        ],
        "all": [
            # Include all optional dependencies
            "qdrant-client>=1.7.0",
            "openai>=1.0.0",
            "cohere>=4.0.0",
            "sentence-transformers>=2.2.0",
            "jsonschema>=4.20.0",
            "httpx>=0.25.0",
            "networkx>=3.0",
            "scikit-learn>=1.3.0",
            "opentelemetry-api>=1.20.0",
            "opentelemetry-sdk>=1.20.0",
            "opentelemetry-instrumentation-fastapi>=0.41b0",
            "anthropic>=0.7.0",
            "google-generativeai>=0.3.0",
            "mistralai>=0.0.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "agentic=agentic_community.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "agentic_community": [
            "py.typed",
            "templates/*.html",
            "static/*.css",
        ],
    },
    zip_safe=False,
)
