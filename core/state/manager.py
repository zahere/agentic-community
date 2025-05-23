"""
State Management System
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import json
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AgentState(BaseModel):
    """Agent state model."""
    agent_id: str = Field(description="Unique agent identifier")
    current_task: Optional[str] = Field(default=None, description="Current task")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Execution history")
    context: Dict[str, Any] = Field(default_factory=dict, description="Current context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class StateManager:
    """Manages agent state and history."""
    
    def __init__(self, persistence_path: Optional[Path] = None):
        """
        Initialize state manager.
        
        Args:
            persistence_path: Optional path for persisting state
        """
        self.states: Dict[str, AgentState] = {}
        self.persistence_path = persistence_path
        
        if persistence_path and persistence_path.exists():
            self._load_states()
            
    def create_state(self, agent_id: str) -> AgentState:
        """Create a new agent state."""
        state = AgentState(agent_id=agent_id)
        self.states[agent_id] = state
        logger.debug(f"Created state for agent: {agent_id}")
        return state
        
    def get_state(self, agent_id: str) -> Optional[AgentState]:
        """Get agent state."""
        return self.states.get(agent_id)
        
    def update_state(self, agent_id: str, updates: Dict[str, Any]) -> AgentState:
        """Update agent state."""
        state = self.states.get(agent_id)
        if not state:
            state = self.create_state(agent_id)
            
        # Update fields
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
                
        state.updated_at = datetime.now()
        
        # Persist if configured
        if self.persistence_path:
            self._save_states()
            
        logger.debug(f"Updated state for agent: {agent_id}")
        return state
        
    def add_to_history(self, agent_id: str, entry: Dict[str, Any]) -> None:
        """Add entry to agent history."""
        state = self.get_state(agent_id)
        if not state:
            state = self.create_state(agent_id)
            
        entry["timestamp"] = datetime.now().isoformat()
        state.history.append(entry)
        state.updated_at = datetime.now()
        
        if self.persistence_path:
            self._save_states()
            
    def get_history(self, agent_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get agent history."""
        state = self.get_state(agent_id)
        if not state:
            return []
            
        history = state.history
        if limit:
            history = history[-limit:]
            
        return history
        
    def clear_history(self, agent_id: str) -> None:
        """Clear agent history."""
        state = self.get_state(agent_id)
        if state:
            state.history = []
            state.updated_at = datetime.now()
            
            if self.persistence_path:
                self._save_states()
                
    def _save_states(self) -> None:
        """Save states to persistence."""
        if not self.persistence_path:
            return
            
        data = {
            agent_id: state.model_dump(mode="json")
            for agent_id, state in self.states.items()
        }
        
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persistence_path, "w") as f:
            json.dump(data, f, indent=2)
            
    def _load_states(self) -> None:
        """Load states from persistence."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
            
        try:
            with open(self.persistence_path, "r") as f:
                data = json.load(f)
                
            for agent_id, state_data in data.items():
                # Convert string dates back to datetime
                for date_field in ["created_at", "updated_at"]:
                    if date_field in state_data:
                        state_data[date_field] = datetime.fromisoformat(state_data[date_field])
                        
                self.states[agent_id] = AgentState(**state_data)
                
        except Exception as e:
            logger.error(f"Failed to load states: {e}")
