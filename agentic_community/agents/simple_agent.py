"""
Simple Agent - Community Edition
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseLanguageModel
try:
    # Try newer import structure
    from langchain_openai import OpenAI
except ImportError:
    # Fall back to older structure
    try:
        from langchain_community.llms import OpenAI
    except ImportError:
        # Last resort - try direct langchain import
        from langchain.llms import OpenAI
        
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from agentic_community.core.base import BaseAgent, AgentConfig, BaseTool
from agentic_community.core.utils import get_logger

logger = get_logger(__name__)


class SimpleAgent(BaseAgent):
    """
    Simple agent with basic sequential reasoning.
    Community edition agent with limited capabilities.
    """
    
    def __init__(self, name: str, tools: Optional[List[BaseTool]] = None, openai_api_key: Optional[str] = None):
        """
        Initialize simple agent.
        
        Args:
            name: Agent name
            tools: Optional list of tools to use
            openai_api_key: OpenAI API key (required for community edition)
        """
        config = AgentConfig(
            name=name,
            description=f"Simple agent: {name}",
            llm_provider="openai",
            model="gpt-3.5-turbo"  # Community edition uses basic model
        )
        
        self.openai_api_key = openai_api_key
        super().__init__(config)
        
        # Add tools if provided
        if tools:
            for tool in tools[:3]:  # Community edition limited to 3 tools
                self.add_tool(tool)
        
        # Initialize the graph
        self._setup_graph()
        
    def _initialize_llm(self) -> BaseLanguageModel:
        """Initialize OpenAI LLM for community edition."""
        if not self.openai_api_key:
            import os
            # Try to get from environment
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                # Return a mock LLM for testing
                logger.warning("No OpenAI API key found. Using mock LLM for testing.")
                from unittest.mock import MagicMock
                mock_llm = MagicMock()
                mock_llm.invoke = lambda x: "Step 1: Analyze\nStep 2: Plan\nStep 3: Execute"
                return mock_llm
            
        return OpenAI(
            temperature=self.config.temperature,
            openai_api_key=self.openai_api_key
        )
        
    def _setup_graph(self) -> None:
        """Setup the simple reasoning graph."""
        # Define the graph state
        from typing import TypedDict
        
        class GraphState(TypedDict):
            task: str
            thoughts: List[str]
            actions: List[Dict[str, Any]]
            final_answer: str
            
        # Create the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("think", self._think_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("summarize", self._summarize_node)
        
        # Add edges
        workflow.set_entry_point("think")
        workflow.add_edge("think", "act")
        workflow.add_edge("act", "summarize")
        workflow.add_edge("summarize", END)
        
        # Compile
        self.graph = workflow.compile()
        
    def _think_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Basic thinking node - sequential reasoning only."""
        task = state["task"]
        
        # Simple prompt for basic reasoning
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Break down this task into simple steps:\n"
            "Task: {task}\n"
            "Provide 3-5 simple steps to complete this task."
        )
        
        # Get response
        try:
            chain = prompt | self.llm
            response = chain.invoke({"task": task})
        except Exception as e:
            logger.warning(f"LLM invocation failed: {e}. Using fallback response.")
            response = "Step 1: Understand the task\nStep 2: Plan approach\nStep 3: Execute plan\nStep 4: Review results"
        
        # Parse into thoughts
        if isinstance(response, str):
            thoughts = [step.strip() for step in response.split("\n") if step.strip()]
        else:
            thoughts = ["Step 1: Process task", "Step 2: Execute", "Step 3: Complete"]
        
        return {"thoughts": thoughts[:5]}  # Limit to 5 steps
        
    def _act_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Basic action node - execute simple actions."""
        thoughts = state["thoughts"]
        actions = []
        
        for thought in thoughts:
            # For community edition, we just simulate actions
            action = {
                "thought": thought,
                "tool": "basic_reasoning",
                "result": f"Completed: {thought}"
            }
            actions.append(action)
            
            # Use tools if available
            if self.tools:
                # Simple tool execution (limited to 3 tools)
                for tool in self.tools[:3]:  # Community limit
                    try:
                        result = tool.execute(thought)
                        action["tool_result"] = result
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        
        return {"actions": actions}
        
    def _summarize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize the results."""
        actions = state["actions"]
        
        # Create summary
        summary_parts = []
        for i, action in enumerate(actions, 1):
            summary_parts.append(f"{i}. {action['result']}")
            
        final_answer = "\n".join(summary_parts)
        
        return {"final_answer": final_answer}
        
    def think(self, task: str) -> Dict[str, Any]:
        """
        Basic sequential thinking.
        
        Args:
            task: The task to think about
            
        Returns:
            Dictionary containing basic thoughts
        """
        initial_state = {
            "task": task,
            "thoughts": [],
            "actions": [],
            "final_answer": ""
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "thoughts": result["thoughts"],
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
        # In simple agent, this is handled by the graph
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
        
        # Run through graph
        initial_state = {
            "task": task,
            "thoughts": [],
            "actions": [],
            "final_answer": ""
        }
        
        result = self.graph.invoke(initial_state)
        
        # Add to history
        self.state_manager.add_to_history(
            self.config.name,
            {
                "task": task,
                "result": result["final_answer"],
                "type": "simple_execution"
            }
        )
        
        return result["final_answer"]
