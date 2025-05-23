"""
Basic Search Tool - Community Edition
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

from agentic_community.core.base import BaseTool
from agentic_community.core.utils import get_logger

logger = get_logger(__name__)


class SearchTool(BaseTool):
    """Basic web search tool for community edition."""
    
    def __init__(self):
        """Initialize search tool."""
        super().__init__(
            name="search",
            description="Search the web for information",
            keywords=["search", "find", "look for", "query", "lookup"]
        )
        
    def run(self, input_text: str) -> str:
        """
        Execute a web search.
        
        Note: In community edition, this is a simulated search.
        For real search, users need to implement their own or upgrade to enterprise.
        
        Args:
            input_text: Search query
            
        Returns:
            Search results as string
        """
        logger.info(f"Searching for: {input_text}")
        
        # Simulated search for community edition
        # Real implementation would require API keys
        results = [
            f"Result 1: Information about {input_text}",
            f"Result 2: Additional details on {input_text}",
            f"Result 3: Related topics to {input_text}"
        ]
        
        return "\n".join(results)
