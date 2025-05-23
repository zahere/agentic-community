"""
Base Tool Class - Foundation for all tools in the framework
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agentic_community.core.utils.logger import get_logger

logger = get_logger(__name__)


class ToolConfig(BaseModel):
    """Configuration for tools."""
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    keywords: List[str] = Field(default_factory=list, description="Keywords that trigger this tool")
    max_retries: int = Field(default=3, description="Max retries on failure")


class BaseTool(ABC):
    """Base tool class for the framework."""
    
    def __init__(self, name: str, description: str = "", keywords: Optional[List[str]] = None):
        """
        Initialize the tool.
        
        Args:
            name: Tool name
            description: Tool description
            keywords: Keywords that trigger this tool
        """
        self.name = name
        self.description = description or f"{name} tool"
        self.keywords = keywords or [name.lower()]
        
        logger.info(f"Initialized tool: {name}")
        
    @abstractmethod
    def run(self, input_text: str) -> str:
        """
        Execute the tool's main functionality.
        Must be implemented by all tools.
        
        Args:
            input_text: Input text to process
            
        Returns:
            Result as string
        """
        pass
        
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool - wrapper for run method."""
        if args:
            return self.run(args[0])
        elif "input_text" in kwargs:
            return self.run(kwargs["input_text"])
        else:
            return self.run("")
            
    def __call__(self, *args, **kwargs) -> Any:
        """Make the tool callable."""
        return self.execute(*args, **kwargs)
        
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"
