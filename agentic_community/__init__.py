"""
Agentic AI Framework - Community Edition

A powerful framework for building autonomous AI agents.
"""

__version__ = "0.1.0"

# Re-export main components
from agentic_community.agents import SimpleAgent
from agentic_community.tools import SearchTool, CalculatorTool, TextTool

__all__ = ["SimpleAgent", "SearchTool", "CalculatorTool", "TextTool"]
