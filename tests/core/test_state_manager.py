"""
Tests for state management functionality
"""

import pytest
import json
from datetime import datetime
from pathlib import Path

from agentic_community.core.state import StateManager, AgentState


class TestAgentState:
    """Test AgentState model."""
    
    def test_state_creation(self):
        """Test creating agent state."""
        state = AgentState(agent_id="test-agent")
        
        assert state.agent_id == "test-agent"
        assert state.current_task is None
        assert state.history == []
        assert state.context == {}
        assert state.metadata == {}
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.updated_at, datetime)
        
    def test_state_with_data(self):
        """Test state with initial data."""
        state = AgentState(
            agent_id="test-agent",
            current_task="Test task",
            context={"key": "value"},
            metadata={"version": "1.0"}
        )
        
        assert state.current_task == "Test task"
        assert state.context["key"] == "value"
        assert state.metadata["version"] == "1.0"


class TestStateManager:
    """Test StateManager functionality."""
    
    def test_manager_initialization(self):
        """Test state manager initialization."""
        manager = StateManager()
        
        assert manager.states == {}
        assert manager.persistence_path is None
        
    def test_create_state(self):
        """Test creating new state."""
        manager = StateManager()
        
        state = manager.create_state("agent1")
        
        assert state.agent_id == "agent1"
        assert "agent1" in manager.states
        assert manager.states["agent1"] == state
        
    def test_get_state(self):
        """Test getting agent state."""
        manager = StateManager()
        manager.create_state("agent1")
        
        state = manager.get_state("agent1")
        assert state is not None
        assert state.agent_id == "agent1"
        
        # Non-existent agent
        state = manager.get_state("agent2")
        assert state is None
        
    def test_update_state(self):
        """Test updating agent state."""
        manager = StateManager()
        
        # Update existing state
        manager.create_state("agent1")
        updated = manager.update_state("agent1", {
            "current_task": "New task",
            "context": {"updated": True}
        })
        
        assert updated.current_task == "New task"
        assert updated.context["updated"] is True
        
        # Update non-existent state (should create)
        updated = manager.update_state("agent2", {
            "current_task": "Another task"
        })
        assert updated.agent_id == "agent2"
        assert updated.current_task == "Another task"
        
    def test_history_management(self):
        """Test history functionality."""
        manager = StateManager()
        manager.create_state("agent1")
        
        # Add to history
        manager.add_to_history("agent1", {
            "action": "test",
            "result": "success"
        })
        
        # Get history
        history = manager.get_history("agent1")
        assert len(history) == 1
        assert history[0]["action"] == "test"
        assert history[0]["result"] == "success"
        assert "timestamp" in history[0]
        
        # Add more entries
        for i in range(5):
            manager.add_to_history("agent1", {
                "action": f"action_{i}"
            })
            
        # Get limited history
        history = manager.get_history("agent1", limit=3)
        assert len(history) == 3
        
        # Clear history
        manager.clear_history("agent1")
        history = manager.get_history("agent1")
        assert len(history) == 0
        
    def test_persistence(self, temp_state_file):
        """Test state persistence."""
        # Create manager with persistence
        manager = StateManager(persistence_path=temp_state_file)
        
        # Create and update state
        manager.create_state("agent1")
        manager.update_state("agent1", {
            "current_task": "Persistent task",
            "context": {"data": "value"}
        })
        manager.add_to_history("agent1", {"action": "test"})
        
        # Save states
        manager._save_states()
        
        # Verify file exists
        assert temp_state_file.exists()
        
        # Load in new manager
        new_manager = StateManager(persistence_path=temp_state_file)
        
        # Verify state loaded
        state = new_manager.get_state("agent1")
        assert state is not None
        assert state.current_task == "Persistent task"
        assert state.context["data"] == "value"
        
        history = new_manager.get_history("agent1")
        assert len(history) == 1
        assert history[0]["action"] == "test"
        
    def test_persistence_auto_save(self, temp_state_file):
        """Test automatic saving on updates."""
        manager = StateManager(persistence_path=temp_state_file)
        
        # Create state (should auto-save)
        manager.create_state("agent1")
        manager.update_state("agent1", {"current_task": "Task 1"})
        
        # Load in new manager to verify
        new_manager = StateManager(persistence_path=temp_state_file)
        state = new_manager.get_state("agent1")
        assert state.current_task == "Task 1"
