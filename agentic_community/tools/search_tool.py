"""
Basic Search Tool - Community Edition
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

from typing import Type
from pydantic import BaseModel, Field
import requests
import json

from agentic_community.core.base import BaseTool, ToolConfig
from agentic_community.core.utils import get_logger

logger = get_logger(__name__)


class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum results to return")


class SearchTool(BaseTool):
    """Basic web search tool for community edition."""
    
    def __init__(self):
        """Initialize search tool."""
        config = ToolConfig(
            name="search",
            description="Search the web for information",
            verbose=True
        )
        super().__init__(config)
        
    @property
    def input_schema(self) -> Type[BaseModel]:
        """Return input schema."""
        return SearchInput
        
    def execute(self, query: str, max_results: int = 5) -> str:
        """
        Execute a web search.
        
        Note: In community edition, this is a simulated search.
        For real search, users need to implement their own or upgrade to enterprise.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            Search results as string
        """
        logger.info(f"Searching for: {query}")
        
        # Simulated search for community edition
        # Real implementation would require API keys
        results = [
            f"Result 1: Information about {query}",
            f"Result 2: Additional details on {query}",
            f"Result 3: Related topics to {query}"
        ]
        
        return "\n".join(results[:max_results])
