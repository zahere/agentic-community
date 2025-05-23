"""
Tests for community edition tools
"""

import pytest
from unittest.mock import Mock, patch

from agentic_community.tools import SearchTool, CalculatorTool, TextTool


class TestSearchTool:
    """Test SearchTool functionality."""
    
    def test_tool_initialization(self):
        """Test search tool initialization."""
        tool = SearchTool()
        
        assert tool.config.name == "search"
        assert tool.config.description == "Search the web for information"
        assert tool.config.verbose is True
        
    def test_input_schema(self):
        """Test input schema."""
        tool = SearchTool()
        schema = tool.input_schema
        
        # Create instance with valid data
        valid_input = schema(query="test query", max_results=3)
        assert valid_input.query == "test query"
        assert valid_input.max_results == 3
        
        # Test defaults
        default_input = schema(query="test")
        assert default_input.max_results == 5
        
    def test_execute(self):
        """Test search execution."""
        tool = SearchTool()
        
        result = tool.execute("Python programming", max_results=3)
        
        assert isinstance(result, str)
        assert "Python programming" in result
        assert len(result.split("\n")) == 3
        
    def test_simulated_search(self):
        """Test that search is simulated in community edition."""
        tool = SearchTool()
        
        result = tool.execute("test query")
        
        # Should return simulated results
        assert "Result 1:" in result
        assert "Result 2:" in result
        assert "Result 3:" in result
        
    def test_to_langchain_tool(self):
        """Test conversion to LangChain tool."""
        tool = SearchTool()
        lc_tool = tool.to_langchain_tool()
        
        assert lc_tool.name == "search"
        assert lc_tool.description == "Search the web for information"


class TestCalculatorTool:
    """Test CalculatorTool functionality."""
    
    def test_tool_initialization(self):
        """Test calculator tool initialization."""
        tool = CalculatorTool()
        
        assert tool.config.name == "calculator"
        assert tool.config.description == "Perform basic mathematical calculations"
        assert tool.config.return_direct is True
        
    def test_basic_operations(self):
        """Test basic math operations."""
        tool = CalculatorTool()
        
        # Addition
        result = tool.execute("2 + 3")
        assert result == "2 + 3 = 5"
        
        # Subtraction
        result = tool.execute("10 - 4")
        assert result == "10 - 4 = 6"
        
        # Multiplication
        result = tool.execute("6 * 7")
        assert result == "6 * 7 = 42"
        
        # Division
        result = tool.execute("15 / 3")
        assert result == "15 / 3 = 5.0"
        
        # Power
        result = tool.execute("2 ** 8")
        assert result == "2 ** 8 = 256"
        
    def test_complex_expressions(self):
        """Test complex expressions."""
        tool = CalculatorTool()
        
        result = tool.execute("(10 + 5) * 2 - 8 / 4")
        assert "28.0" in result
        
    def test_unsafe_operations(self):
        """Test that unsafe operations are blocked."""
        tool = CalculatorTool()
        
        # Should reject function calls
        result = tool.execute("eval('2+2')")
        assert "Error" in result
        
        # Should reject imports
        result = tool.execute("__import__('os')")
        assert "Error" in result
        
    def test_error_handling(self):
        """Test error handling."""
        tool = CalculatorTool()
        
        # Division by zero
        result = tool.execute("1 / 0")
        assert "Error" in result
        
        # Invalid expression
        result = tool.execute("invalid math")
        assert "Error" in result


class TestTextTool:
    """Test TextTool functionality."""
    
    def test_tool_initialization(self):
        """Test text tool initialization."""
        tool = TextTool()
        
        assert tool.config.name == "text_processor"
        assert tool.config.description == "Process and analyze text with basic operations"
        
    def test_summarize_operation(self):
        """Test text summarization."""
        tool = TextTool()
        
        # Short text
        result = tool.execute("This is a short text.", "summarize")
        assert "Summary:" in result
        assert "This is a short text." in result
        
        # Long text (>100 words)
        long_text = " ".join(["word"] * 150)
        result = tool.execute(long_text, "summarize")
        assert "Summary:" in result
        assert "..." in result
        
    def test_extract_operation(self):
        """Test pattern extraction."""
        tool = TextTool()
        
        # Extract emails
        text = "Contact us at test@example.com or support@test.org"
        result = tool.execute(text, "extract", r"\S+@\S+")
        assert "test@example.com" in result
        assert "support@test.org" in result
        
        # Missing pattern
        result = tool.execute(text, "extract", "")
        assert "Error" in result
        
    def test_count_words_operation(self):
        """Test word counting."""
        tool = TextTool()
        
        text = "This is a test sentence with seven words."
        result = tool.execute(text, "count_words")
        
        assert "Words: 8" in result
        assert "Characters: 42" in result
        
    def test_clean_operation(self):
        """Test text cleaning."""
        tool = TextTool()
        
        # Text with extra whitespace and special characters
        text = "  This   is    messy!!!   @#$%   text...  "
        result = tool.execute(text, "clean")
        
        assert "Cleaned text:" in result
        assert "This is messy!!! text..." in result
        
    def test_unknown_operation(self):
        """Test unknown operation handling."""
        tool = TextTool()
        
        result = tool.execute("test", "unknown_op")
        assert "Unknown operation: unknown_op" in result
        
    def test_error_handling(self):
        """Test error handling."""
        tool = TextTool()
        
        # Invalid regex pattern
        result = tool.execute("test", "extract", "[")
        assert "Error" in result
