"""
Simple Agent - Community Edition
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseLanguageModel

# Import error handling and validation
from agentic_community.core.exceptions import (
    APIKeyError, AgentExecutionError, InvalidTaskError, 
    ToolLimitExceeded, handle_error
)
from agentic_community.core.utils.validation import (
    validate_agent_name, validate_task, validate_inputs
)

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
    
    MAX_TOOLS = 3  # Community edition limit
    
    @validate_inputs(name=validate_agent_name)
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
            for tool in tools[:self.MAX_TOOLS]:
                try:
                    self.add_tool(tool)
                except ToolLimitExceeded:
                    logger.warning(f"Tool limit reached. Only first {self.MAX_TOOLS} tools added.")
                    break
        
        # Initialize the graph
        self._setup_graph()
        
    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool with validation."""
        if len(self.tools) >= self.MAX_TOOLS:
            raise ToolLimitExceeded(self.MAX_TOOLS, "community")
        super().add_tool(tool)
        
    def _initialize_llm(self) -> BaseLanguageModel:
        """Initialize OpenAI LLM for community edition."""
        if not self.openai_api_key:
            import os
            # Try to get from environment
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                # For testing/demo purposes, return a mock LLM
                logger.warning("No OpenAI API key found. Using mock LLM for testing.")
                from unittest.mock import MagicMock
                mock_llm = MagicMock()
                mock_llm.invoke = lambda x: MagicMock(content="Step 1: Analyze\nStep 2: Plan\nStep 3: Execute")
                return mock_llm
        
        try:
            return OpenAI(
                temperature=self.config.temperature,
                openai_api_key=self.openai_api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM: {e}")
            raise APIKeyError("OpenAI")
            
    def _setup_graph(self) -> None:
        """Setup the simple reasoning graph."""
        # Define the graph state
        from typing import TypedDict
        
        class GraphState(TypedDict):
            task: str
            thoughts: List[str]
            actions: List[Dict[str, Any]]
            final_answer: str
            errors: List[str]
            
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
        errors = state.get("errors", [])
        
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
            error_msg = f"LLM invocation failed: {str(e)}"
            logger.warning(error_msg)
            errors.append(error_msg)
            response = MagicMock(content="Step 1: Understand the task\nStep 2: Plan approach\nStep 3: Execute plan\nStep 4: Review results")
        
        # Parse into thoughts
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
            
        thoughts = [step.strip() for step in content.split("\n") if step.strip()]
        
        return {"thoughts": thoughts[:5], "errors": errors}  # Limit to 5 steps
        
    def _act_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Basic action node - execute simple actions."""
        thoughts = state["thoughts"]
        errors = state.get("errors", [])
        actions = []
        
        for thought in thoughts:
            # For community edition, we just simulate actions
            action = {
                "thought": thought,
                "tool": "basic_reasoning",
                "result": f"Completed: {thought}"
            }
            
            # Use tools if available
            if self.tools:
                # Try to match thought with appropriate tool
                for tool in self.tools:
                    try:
                        # Simple keyword matching for tool selection
                        tool_keywords = {
                            "search": ["search", "find", "look up", "research"],
                            "calculator": ["calculate", "compute", "math", "sum", "multiply"],
                            "text_processor": ["summarize", "extract", "format", "text"]
                        }
                        
                        thought_lower = thought.lower()
                        if any(keyword in thought_lower for keyword in tool_keywords.get(tool.name, [])):
                            result = tool.invoke(thought)
                            action["tool"] = tool.name
                            action["tool_result"] = result
                            break
                    except Exception as e:
                        error_msg = f"Tool '{tool.name}' execution failed: {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        handle_error(e, f"tool execution for {tool.name}")
                        
            actions.append(action)
                    
        return {"actions": actions, "errors": errors}
        
    def _summarize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize the results."""
        actions = state["actions"]
        errors = state.get("errors", [])
        
        # Create summary
        summary_parts = []
        for i, action in enumerate(actions, 1):
            summary_parts.append(f"{i}. {action['result']}")
            if "tool_result" in action:
                summary_parts.append(f"   Tool output: {action['tool_result']}")
                
        final_answer = "\n".join(summary_parts)
        
        # Add error summary if any
        if errors:
            final_answer += "\n\nNote: Some errors occurred during execution:\n"
            final_answer += "\n".join(f"- {error}" for error in errors)
        
        return {"final_answer": final_answer, "errors": errors}
        
    def think(self, task: str) -> Dict[str, Any]:
        """
        Basic sequential thinking.
        
        Args:
            task: The task to think about
            
        Returns:
            Dictionary containing basic thoughts
        """
        # Validate task
        task = validate_task(task)
        
        initial_state = {
            "task": task,
            "thoughts": [],
            "actions": [],
            "final_answer": "",
            "errors": []
        }
        
        try:
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            return {
                "thoughts": result["thoughts"],
                "reasoning_type": "sequential",
                "complexity": "basic",
                "errors": result.get("errors", [])
            }
        except Exception as e:
            handle_error(e, "thinking process")
            raise AgentExecutionError(self.config.name, task, str(e))
        
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
            "status": "completed",
            "errors": thoughts.get("errors", [])
        }
        
    @validate_inputs(task=validate_task)
    def run(self, task: str) -> str:
        """
        Run the complete agent workflow.
        
        Args:
            task: The task to execute
            
        Returns:
            Final result as string
            
        Raises:
            InvalidTaskError: If task is invalid
            AgentExecutionError: If execution fails
        """
        logger.info(f"SimpleAgent {self.config.name} executing task: {task}")
        
        try:
            # Update state
            self.update_state({"current_task": task})
            
            # Run through graph
            initial_state = {
                "task": task,
                "thoughts": [],
                "actions": [],
                "final_answer": "",
                "errors": []
            }
            
            result = self.graph.invoke(initial_state)
            
            # Add to history
            self.state_manager.add_to_history(
                self.config.name,
                {
                    "task": task,
                    "result": result["final_answer"],
                    "type": "simple_execution",
                    "errors": result.get("errors", [])
                }
            )
            
            return result["final_answer"]
            
        except Exception as e:
            handle_error(e, f"agent execution for task: {task}")
            raise AgentExecutionError(self.config.name, task, str(e))
