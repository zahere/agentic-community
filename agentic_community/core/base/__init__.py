"""Base module - Shared components between editions."""

from .agent import BaseAgent, AgentConfig
from .tool import BaseTool

__all__ = ["BaseAgent", "AgentConfig", "BaseTool"]
