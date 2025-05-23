"""
Agentic AI Framework - Community Edition
Free for commercial and non-commercial use
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

__version__ = "1.0.0"

from .agents import SimpleAgent
from .tools import SearchTool, CalculatorTool, TextTool

__all__ = [
    "SimpleAgent",
    "SearchTool",
    "CalculatorTool",
    "TextTool"
]

# Optional: Import API if running as service
try:
    from .api import app
    __all__.append("app")
except ImportError:
    pass  # API dependencies not installed
