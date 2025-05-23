"""
Base Tool Class - Foundation for all tools in the framework
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool as LangChainBaseTool

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ToolConfig(BaseModel):
    """Configuration for tools."""
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    return_direct: bool = Field(default=False, description="Return result directly")
    verbose: bool = Field(default=False, description="Verbose logging")
    max_retries: int = Field(default=3, description="Max retries on failure")


class BaseTool(ABC):
    """Base tool class for the framework."""
    
    def __init__(self, config: ToolConfig):
        """Initialize the tool with configuration."""
        self.config = config
        self._langchain_tool: Optional[LangChainBaseTool] = None
        logger.info(f"Initializing tool: {config.name}")
        
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the tool's main functionality.
        Must be implemented by all tools.
        """
        pass
        
    @property
    @abstractmethod
    def input_schema(self) -> Type[BaseModel]:
        """Define the input schema for the tool."""
        pass
        
    def to_langchain_tool(self) -> LangChainBaseTool:
        """Convert to LangChain compatible tool."""
        if not self._langchain_tool:
            from langchain_core.tools import Tool
            
            self._langchain_tool = Tool(
                name=self.config.name,
                description=self.config.description,
                func=self.execute,
                return_direct=self.config.return_direct,
            )
            
        return self._langchain_tool
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input against schema."""
        try:
            self.input_schema(**input_data)
            return True
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
            
    def __call__(self, *args, **kwargs) -> Any:
        """Make the tool callable."""
        return self.execute(*args, **kwargs)
        
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.config.name})"
