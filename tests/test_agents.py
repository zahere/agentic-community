"""Tests for agent functionality."""

import pytest
from unittest.mock import MagicMock, patch
from agentic_community import SimpleAgent
from agentic_community.tools import SearchTool, CalculatorTool, TextTool


class TestSimpleAgent:
    """Test SimpleAgent functionality."""
    
    def test_agent_creation(self):
        """Test basic agent creation."""
        agent = SimpleAgent("TestAgent")
        assert agent.name == "TestAgent"
        assert len(agent.tools) == 0
    
    def test_agent_with_tools(self):
        """Test agent creation with tools."""
        tools = [SearchTool(), CalculatorTool()]
        agent = SimpleAgent("TestAgent", tools=tools)
        assert len(agent.tools) == 2
    
    def test_add_tool(self):
        """Test adding tools to agent."""
        agent = SimpleAgent("TestAgent")
        agent.add_tool(SearchTool())
        assert len(agent.tools) == 1
        
        agent.add_tool(CalculatorTool())
        assert len(agent.tools) == 2
    
    def test_tool_limit(self):
        """Test that community edition enforces tool limit."""
        agent = SimpleAgent("TestAgent")
        agent.add_tool(SearchTool())
        agent.add_tool(CalculatorTool())
        agent.add_tool(TextTool())
        
        # Fourth tool should raise exception
        with pytest.raises(Exception):
            agent.add_tool(SearchTool())
    
    @patch('os.environ.get')
    def test_run_simple_task(self, mock_env):
        """Test running a simple task."""
        mock_env.return_value = "mock-api-key"
        
        agent = SimpleAgent("TestAgent")
        agent.add_tool(CalculatorTool())
        
        # Mock the LLM response
        with patch.object(agent, '_execute') as mock_execute:
            mock_execute.return_value = "The result is 5"
            result = agent.run("What is 2 + 3?")
            assert "5" in result
    
    def test_get_state(self):
        """Test getting agent state."""
        agent = SimpleAgent("TestAgent")
        agent.add_tool(SearchTool())
        
        state = agent.get_state()
        assert state["name"] == "TestAgent"
        assert "tools" in state
        assert len(state["tools"]) == 1
    
    def test_set_state(self):
        """Test setting agent state."""
        agent = SimpleAgent("TestAgent")
        
        state = {
            "name": "TestAgent",
            "tools": ["search"],
            "history": [{"task": "test", "result": "test result"}]
        }
        
        agent.set_state(state)
        new_state = agent.get_state()
        assert new_state["name"] == "TestAgent"
    
    def test_clear_history(self):
        """Test clearing agent history."""
        agent = SimpleAgent("TestAgent")
        
        # Add some history
        agent._history = [{"task": "test", "result": "result"}]
        
        agent.clear_history()
        state = agent.get_state()
        assert len(state.get("history", [])) == 0
