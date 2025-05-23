"""
Basic Text Tool - Community Edition
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

from typing import Type, Literal
from pydantic import BaseModel, Field
import re

from agentic_community.core.base import BaseTool, ToolConfig
from agentic_community.core.utils import get_logger

logger = get_logger(__name__)


class TextInput(BaseModel):
    """Input schema for text tool."""
    text: str = Field(description="Text to process")
    operation: Literal["summarize", "extract", "count_words", "clean"] = Field(
        description="Operation to perform"
    )
    pattern: str = Field(default="", description="Pattern for extraction (regex)")


class TextTool(BaseTool):
    """Basic text processing tool for community edition."""
    
    def __init__(self):
        """Initialize text tool."""
        config = ToolConfig(
            name="text_processor",
            description="Process and analyze text with basic operations"
        )
        super().__init__(config)
        
    @property
    def input_schema(self) -> Type[BaseModel]:
        """Return input schema."""
        return TextInput
        
    def execute(self, text: str, operation: str, pattern: str = "") -> str:
        """
        Execute text processing operation.
        
        Args:
            text: Text to process
            operation: Operation to perform
            pattern: Optional pattern for extraction
            
        Returns:
            Processed text result
        """
        logger.info(f"Processing text with operation: {operation}")
        
        try:
            if operation == "summarize":
                # Basic summarization - first 100 words
                words = text.split()
                summary = " ".join(words[:100])
                if len(words) > 100:
                    summary += "..."
                return f"Summary: {summary}"
                
            elif operation == "extract":
                # Extract using pattern
                if not pattern:
                    return "Error: Pattern required for extraction"
                    
                matches = re.findall(pattern, text)
                return f"Extracted: {', '.join(matches)}"
                
            elif operation == "count_words":
                # Count words
                word_count = len(text.split())
                char_count = len(text)
                return f"Words: {word_count}, Characters: {char_count}"
                
            elif operation == "clean":
                # Basic text cleaning
                # Remove extra whitespace
                cleaned = " ".join(text.split())
                # Remove special characters (keep alphanumeric and basic punctuation)
                cleaned = re.sub(r'[^\w\s.,!?-]', '', cleaned)
                return f"Cleaned text: {cleaned}"
                
            else:
                return f"Unknown operation: {operation}"
                
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return f"Error processing text: {str(e)}"
