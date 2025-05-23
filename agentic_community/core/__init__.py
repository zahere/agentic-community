"""
Agentic AI Framework - Core Components
Shared between Community and Enterprise editions
"""

__version__ = "1.0.0"

from .base.agent import BaseAgent
from .base.tool import BaseTool
from .state.manager import StateManager

__all__ = ["BaseAgent", "BaseTool", "StateManager"]
