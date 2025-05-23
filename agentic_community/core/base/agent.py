"""
Base Agent Class - Shared between Community and Enterprise Editions
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from agentic_community.core.state.manager import StateManager
from agentic_community.core.utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Base agent class for both Community and Enterprise editions."""
    
    def __init__(self, name: str, tools: Optional[List[Any]] = None):
        """
        Initialize the agent.
        
        Args:
            name: Agent name
            tools: List of available tools
        """
        self.name = name
        self.state_manager = StateManager()
        self.tools = tools or []
        
        # Initialize state
        self.state_manager.update_state("name", name)
        self.state_manager.update_state("status", "initialized")
        
        logger.info(f"Initialized agent: {name}")
        
    @abstractmethod
    def execute(self, task: str) -> str:
        """
        Execute a task.
        
        Args:
            task: The task to execute
            
        Returns:
            Result of task execution
        """
        pass
        
    def add_tool(self, tool: Any) -> None:
        """Add a tool to the agent."""
        self.tools.append(tool)
        logger.debug(f"Added tool {tool.name} to agent {self.name}")
        
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        return {
            "name": self.name,
            "status": self.state_manager.get_state().get("status", "unknown"),
            "tool_count": len(self.tools),
            "state": self.state_manager.get_state()
        }
        
    def reset(self):
        """Reset agent state."""
        self.state_manager.reset()
        self.state_manager.update_state("name", self.name)
        self.state_manager.update_state("status", "initialized")
        logger.info(f"Agent {self.name} state reset")
