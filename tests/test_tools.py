"""Tests for tool functionality."""

import pytest
import math
from agentic_community.tools import SearchTool, CalculatorTool, TextTool


class TestSearchTool:
    """Test SearchTool functionality."""
    
    def test_search_tool_creation(self):
        """Test creating a search tool."""
        tool = SearchTool()
        assert tool.name == "search"
        assert "search" in tool.description.lower()
    
    def test_search_tool_mock(self):
        """Test that search tool returns mock results."""
        tool = SearchTool()
        # In community edition, this is mocked
        result = tool.search("Python programming")
        assert isinstance(result, str)
        assert len(result) > 0


class TestCalculatorTool:
    """Test CalculatorTool functionality."""
    
    def test_calculator_creation(self):
        """Test creating a calculator tool."""
        tool = CalculatorTool()
        assert tool.name == "calculator"
        assert "math" in tool.description.lower() or "calc" in tool.description.lower()
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        tool = CalculatorTool()
        
        assert tool.calculate("2 + 2") == 4
        assert tool.calculate("10 - 5") == 5
        assert tool.calculate("3 * 4") == 12
        assert tool.calculate("15 / 3") == 5
        assert tool.calculate("2 ** 3") == 8
        assert tool.calculate("10 % 3") == 1
    
    def test_advanced_operations(self):
        """Test advanced mathematical operations."""
        tool = CalculatorTool()
        
        # Test with some tolerance for floating point
        assert abs(tool.calculate("sqrt(16)") - 4) < 0.001
        assert abs(tool.calculate("sin(0)") - 0) < 0.001
        assert abs(tool.calculate("cos(0)") - 1) < 0.001
        assert abs(tool.calculate("log(10)") - math.log(10)) < 0.001
    
    def test_invalid_expression(self):
        """Test handling of invalid expressions."""
        tool = CalculatorTool()
        
        with pytest.raises(Exception):
            tool.calculate("invalid expression")
        
        with pytest.raises(Exception):
            tool.calculate("2 ++ 2")


class TestTextTool:
    """Test TextTool functionality."""
    
    def test_text_tool_creation(self):
        """Test creating a text tool."""
        tool = TextTool()
        assert tool.name == "text_processor"
        assert "text" in tool.description.lower()
    
    def test_word_count(self):
        """Test word counting functionality."""
        tool = TextTool()
        text = "This is a simple test sentence."
        
        result = tool.process(text, operation="count_words")
        assert result == 6
    
    def test_character_count(self):
        """Test character counting functionality."""
        tool = TextTool()
        text = "Hello World"
        
        result = tool.process(text, operation="count_chars")
        assert result == 11  # Including space
    
    def test_summarize(self):
        """Test text summarization (mock in community edition)."""
        tool = TextTool()
        long_text = "This is a very long text. " * 20
        
        result = tool.process(long_text, operation="summarize")
        assert isinstance(result, str)
        assert len(result) < len(long_text)
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        tool = TextTool()
        text = "Python programming is great for data science and machine learning."
        
        result = tool.process(text, operation="extract_keywords")
        assert isinstance(result, list)
        assert len(result) > 0
