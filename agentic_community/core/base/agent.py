"""
Base Agent Class - Shared between Community and Enterprise Editions
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph

from ..state.manager import StateManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AgentConfig(BaseModel):
    """Configuration for agents."""
    name: str = Field(description="Agent name")
    description: str = Field(default="", description="Agent description")
    llm_provider: str = Field(default="openai", description="LLM provider")
    model: str = Field(default="gpt-4", description="Model name")
    temperature: float = Field(default=0.7, description="Temperature for LLM")
    max_iterations: int = Field(default=10, description="Max reasoning iterations")
    verbose: bool = Field(default=False, description="Verbose logging")


class BaseAgent(ABC):
    """Base agent class for both Community and Enterprise editions."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the agent with configuration."""
        self.config = config
        self.state_manager = StateManager()
        self.llm = self._initialize_llm()
        self.tools: List[Any] = []
        self.graph: Optional[StateGraph] = None
        
        # Initialize state for this agent
        self.state_manager.create_state(config.name)
        
        logger.info(f"Initializing agent: {config.name}")
        
    def _initialize_llm(self) -> BaseLanguageModel:
        """Initialize the language model."""
        # This will be overridden by specific implementations
        # Community edition might only support OpenAI
        # Enterprise edition supports multiple providers
        raise NotImplementedError("Subclasses must implement LLM initialization")
        
    @abstractmethod
    def think(self, task: str) -> Dict[str, Any]:
        """
        Core thinking method - must be implemented by subclasses.
        
        Args:
            task: The task to think about
            
        Returns:
            Dictionary containing thoughts and reasoning
        """
        pass
        
    @abstractmethod
    def act(self, thoughts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute actions based on thoughts.
        
        Args:
            thoughts: The thoughts from the thinking phase
            
        Returns:
            Dictionary containing action results
        """
        pass
        
    def run(self, task: str) -> str:
        """
        Main execution method - think then act.
        
        Args:
            task: The task to execute
            
        Returns:
            Final result as string
        """
        logger.info(f"Agent {self.config.name} running task: {task}")
        
        # Think
        thoughts = self.think(task)
        
        # Act
        results = self.act(thoughts)
        
        # Format response
        return self._format_response(results)
        
    def _format_response(self, results: Dict[str, Any]) -> str:
        """Format the final response."""
        # Default implementation - can be overridden
        if "output" in results:
            return results["output"]
        return str(results)
        
    def add_tool(self, tool: Any) -> None:
        """Add a tool to the agent."""
        self.tools.append(tool)
        logger.debug(f"Added tool to agent {self.config.name}")
        
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        state = self.state_manager.get_state(self.config.name)
        if state:
            return state.model_dump()
        return {}
        
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update agent state."""
        self.state_manager.update_state(self.config.name, updates)
