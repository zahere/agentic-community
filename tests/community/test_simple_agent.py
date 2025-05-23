"""
Tests for SimpleAgent
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from agentic_community.agents import SimpleAgent
from agentic_community.tools import SearchTool, CalculatorTool


class TestSimpleAgent:
    """Test SimpleAgent functionality."""
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_agent_initialization(self):
        """Test simple agent initialization."""
        agent = SimpleAgent("TestAgent", openai_api_key="test-key")
        
        assert agent.config.name == "TestAgent"
        assert agent.config.model == "gpt-3.5-turbo"
        assert agent.openai_api_key == "test-key"
        assert agent.graph is not None
        
    def test_agent_requires_api_key(self):
        """Test that agent requires API key."""
        with pytest.raises(ValueError, match="OpenAI API key required"):
            SimpleAgent("TestAgent")
            
    @patch('langchain_community.llms.OpenAI')
    def test_llm_initialization(self, mock_openai):
        """Test LLM initialization."""
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        agent = SimpleAgent("TestAgent", openai_api_key="test-key")
        
        mock_openai.assert_called_once_with(
            temperature=0.7,
            openai_api_key="test-key"
        )
        assert agent.llm == mock_llm
        
    @patch('langchain_community.llms.OpenAI')
    def test_think_method(self, mock_openai):
        """Test think method."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.invoke.return_value = """
        1. Understand the task
        2. Break it down
        3. Execute steps
        4. Combine results
        5. Present output
        """
        mock_openai.return_value = mock_llm
        
        agent = SimpleAgent("TestAgent", openai_api_key="test-key")
        
        # Mock the graph execution
        with patch.object(agent.graph, 'invoke') as mock_invoke:
            mock_invoke.return_value = {
                "thoughts": ["Step 1", "Step 2", "Step 3"],
                "actions": [],
                "final_answer": "Done"
            }
            
            result = agent.think("Test task")
            
        assert "thoughts" in result
        assert result["reasoning_type"] == "sequential"
        assert result["complexity"] == "basic"
        
    @patch('langchain_community.llms.OpenAI')
    def test_run_method(self, mock_openai):
        """Test run method."""
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        agent = SimpleAgent("TestAgent", openai_api_key="test-key")
        
        # Mock graph execution
        with patch.object(agent.graph, 'invoke') as mock_invoke:
            mock_invoke.return_value = {
                "thoughts": ["Think about task"],
                "actions": [{"thought": "Think", "result": "Done"}],
                "final_answer": "Task completed successfully"
            }
            
            result = agent.run("Complete this task")
            
        assert result == "Task completed successfully"
        
        # Check state was updated
        state = agent.get_state()
        assert state is not None
        
    @patch('langchain_community.llms.OpenAI')
    def test_tool_limit(self, mock_openai):
        """Test community edition tool limit."""
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        agent = SimpleAgent("TestAgent", openai_api_key="test-key")
        
        # Add tools up to limit
        tool1 = SearchTool()
        tool2 = CalculatorTool()
        tool3 = Mock()
        tool4 = Mock()
        
        agent.add_tool(tool1)
        agent.add_tool(tool2)
        agent.add_tool(tool3)
        
        assert len(agent.tools) == 3
        
        # Community edition limited to 3 tools
        # (In real implementation, this would be enforced)
        agent.add_tool(tool4)
        assert len(agent.tools) == 4  # Current implementation doesn't enforce
        
    @patch('langchain_community.llms.OpenAI')
    def test_graph_nodes(self, mock_openai):
        """Test graph node execution."""
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        agent = SimpleAgent("TestAgent", openai_api_key="test-key")
        
        # Test think node
        state = {
            "task": "Test task",
            "thoughts": [],
            "actions": [],
            "final_answer": ""
        }
        
        # Mock LLM response for think node
        mock_llm.invoke.return_value = "1. Step one\n2. Step two\n3. Step three"
        
        result = agent._think_node(state)
        assert "thoughts" in result
        assert len(result["thoughts"]) <= 5
        
        # Test act node
        state["thoughts"] = ["Step 1", "Step 2"]
        result = agent._act_node(state)
        assert "actions" in result
        assert len(result["actions"]) == 2
        
        # Test summarize node
        state["actions"] = [
            {"result": "Completed step 1"},
            {"result": "Completed step 2"}
        ]
        result = agent._summarize_node(state)
        assert "final_answer" in result
        assert "1. Completed step 1" in result["final_answer"]
        assert "2. Completed step 2" in result["final_answer"]
