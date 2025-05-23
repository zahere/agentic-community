"""
Tests for base agent functionality
"""

import pytest
from unittest.mock import Mock, patch

from agentic_community.core.base import BaseAgent, AgentConfig
from agentic_community.core.state import StateManager


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent."""
    
    def _initialize_llm(self):
        """Return mock LLM."""
        return Mock()
        
    def think(self, task: str):
        """Simple think implementation."""
        return {"thoughts": [f"Thinking about: {task}"]}
        
    def act(self, thoughts):
        """Simple act implementation."""
        return {"output": "Action completed"}


class TestBaseAgent:
    """Test BaseAgent functionality."""
    
    def test_agent_initialization(self):
        """Test agent can be initialized."""
        config = AgentConfig(
            name="TestAgent",
            description="Test agent"
        )
        agent = TestAgent(config)
        
        assert agent.config.name == "TestAgent"
        assert agent.config.description == "Test agent"
        assert isinstance(agent.state_manager, StateManager)
        assert agent.tools == []
        
    def test_agent_run(self):
        """Test agent run method."""
        config = AgentConfig(name="TestAgent")
        agent = TestAgent(config)
        
        result = agent.run("Test task")
        
        assert result == "Action completed"
        
    def test_add_tool(self):
        """Test adding tools to agent."""
        config = AgentConfig(name="TestAgent")
        agent = TestAgent(config)
        
        mock_tool = Mock()
        agent.add_tool(mock_tool)
        
        assert len(agent.tools) == 1
        assert agent.tools[0] == mock_tool
        
    def test_state_management(self):
        """Test agent state management."""
        config = AgentConfig(name="TestAgent")
        agent = TestAgent(config)
        
        # Update state
        agent.update_state({"status": "active"})
        
        # Get state
        state = agent.get_state()
        assert state is not None
        
    def test_config_defaults(self):
        """Test AgentConfig default values."""
        config = AgentConfig(name="TestAgent")
        
        assert config.name == "TestAgent"
        assert config.description == ""
        assert config.llm_provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_iterations == 10
        assert config.verbose is False


class TestAgentConfig:
    """Test AgentConfig model."""
    
    def test_config_validation(self):
        """Test config validation."""
        # Valid config
        config = AgentConfig(
            name="ValidAgent",
            temperature=0.5,
            max_iterations=5
        )
        assert config.temperature == 0.5
        assert config.max_iterations == 5
        
    def test_config_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValueError):
            # Missing required name
            AgentConfig()
