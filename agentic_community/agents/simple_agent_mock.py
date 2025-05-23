"""
Mock Simple Agent for testing without LangChain dependencies
This is a temporary solution for testing the framework structure.
"""

from typing import Any, Dict, List, Optional
from agentic_community.core.base import BaseAgent, AgentConfig
from agentic_community.core.utils import get_logger

logger = get_logger(__name__)


class MockLLM:
    """Mock LLM for testing without OpenAI"""
    def __init__(self, **kwargs):
        self.config = kwargs
        
    def invoke(self, prompt):
        """Mock response"""
        return "Step 1: Analyze the task\nStep 2: Plan the approach\nStep 3: Execute the plan\nStep 4: Review results"


class SimpleAgent(BaseAgent):
    """
    Simple agent with basic sequential reasoning.
    Community edition agent with limited capabilities.
    (Mock version for testing)
    """
    
    def __init__(self, name: str, openai_api_key: Optional[str] = None):
        """
        Initialize simple agent.
        
        Args:
            name: Agent name
            openai_api_key: OpenAI API key (ignored in mock)
        """
        config = AgentConfig(
            name=name,
            description=f"Simple agent: {name}",
            llm_provider="mock",
            model="mock-model"
        )
        
        self.openai_api_key = openai_api_key
        super().__init__(config)
        
    def _initialize_llm(self):
        """Initialize Mock LLM for testing."""
        logger.warning("Using Mock LLM - for testing only!")
        return MockLLM(temperature=self.config.temperature)
        
    def think(self, task: str) -> Dict[str, Any]:
        """
        Basic sequential thinking.
        
        Args:
            task: The task to think about
            
        Returns:
            Dictionary containing basic thoughts
        """
        # Mock thinking process
        thoughts = [
            f"Understanding task: {task}",
            "Breaking down into steps",
            "Planning execution",
            "Preparing to act"
        ]
        
        return {
            "thoughts": thoughts,
            "reasoning_type": "sequential",
            "complexity": "basic"
        }
        
    def act(self, thoughts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute actions based on thoughts.
        
        Args:
            thoughts: The thoughts from thinking phase
            
        Returns:
            Dictionary containing action results
        """
        return {
            "actions_taken": len(thoughts.get("thoughts", [])),
            "status": "completed"
        }
        
    def run(self, task: str) -> str:
        """
        Run the complete agent workflow.
        
        Args:
            task: The task to execute
            
        Returns:
            Final result as string
        """
        logger.info(f"SimpleAgent {self.config.name} executing task: {task}")
        
        # Update state
        self.update_state({"current_task": task})
        
        # Mock execution
        thoughts = self.think(task)
        actions = self.act(thoughts)
        
        result = f"Task '{task}' completed with {actions['actions_taken']} actions."
        
        # Add to history
        self.state_manager.add_to_history(
            self.config.name,
            {
                "task": task,
                "result": result,
                "type": "simple_execution"
            }
        )
        
        return result
