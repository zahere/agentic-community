"""Base module - Shared components between editions."""

from .agent import BaseAgent
from .tool import BaseTool, ToolConfig

__all__ = ["BaseAgent", "BaseTool", "ToolConfig"]
