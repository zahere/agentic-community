"""
Simple Agent - Community Edition
A lightweight agent implementation without external dependencies.
"""

from typing import Any, Dict, List, Optional
from agentic_community.core.base import BaseAgent
from agentic_community.core.utils import get_logger
from agentic_community.core.reasoning import ReasoningEngine

logger = get_logger(__name__)


class SimpleAgent(BaseAgent):
    """
    Simple agent with basic sequential reasoning.
    Community edition agent without external dependencies.
    """
    
    def __init__(self, name: str, tools: Optional[List[Any]] = None):
        """
        Initialize simple agent.
        
        Args:
            name: Agent name
            tools: List of available tools
        """
        super().__init__(name, tools or [])
        self.reasoning_engine = ReasoningEngine()
        logger.info(f"Initialized SimpleAgent '{name}' with {len(self.tools)} tools")
        
    def execute(self, task: str) -> str:
        """
        Execute a task using simple sequential reasoning.
        
        Args:
            task: The task to execute
            
        Returns:
            Result of task execution
        """
        logger.info(f"Executing task: {task}")
        
        # Update agent state
        self.state_manager.update_state("current_task", task)
        self.state_manager.update_state("status", "thinking")
        
        # Step 1: Analyze the task
        analysis = self._analyze_task(task)
        
        # Step 2: Plan approach
        plan = self._plan_approach(analysis)
        
        # Step 3: Execute plan
        result = self._execute_plan(plan, task)
        
        # Update history
        self.state_manager.add_to_history({
            "task": task,
            "analysis": analysis,
            "plan": plan,
            "result": result,
            "timestamp": self._get_timestamp()
        })
        
        # Update final state
        self.state_manager.update_state("status", "completed")
        
        return result
        
    def _analyze_task(self, task: str) -> Dict[str, Any]:
        """Analyze the task to understand what needs to be done."""
        # Use reasoning engine for basic analysis
        reasoning_result = self.reasoning_engine.process(task)
        
        # Identify required tools
        required_tools = []
        task_lower = task.lower()
        
        for tool in self.tools:
            # Check if tool might be relevant based on keywords
            tool_keywords = getattr(tool, 'keywords', [tool.name.lower()])
            if any(keyword in task_lower for keyword in tool_keywords):
                required_tools.append(tool.name)
                
        analysis = {
            "task_type": self._identify_task_type(task),
            "complexity": "simple",  # Community edition handles simple tasks
            "required_tools": required_tools,
            "reasoning": reasoning_result
        }
        
        logger.debug(f"Task analysis: {analysis}")
        return analysis
        
    def _identify_task_type(self, task: str) -> str:
        """Identify the type of task."""
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["calculate", "compute", "sum", "multiply"]):
            return "calculation"
        elif any(word in task_lower for word in ["search", "find", "look for", "query"]):
            return "search"
        elif any(word in task_lower for word in ["analyze", "examine", "inspect"]):
            return "analysis"
        elif any(word in task_lower for word in ["summarize", "extract", "shorten"]):
            return "summarization"
        else:
            return "general"
            
    def _plan_approach(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan the approach based on task analysis."""
        plan = []
        task_type = analysis["task_type"]
        required_tools = analysis["required_tools"]
        
        # Create simple sequential plan
        if task_type == "calculation":
            plan.append({
                "step": "extract_numbers",
                "description": "Extract numbers and operations from the task",
                "tool": "calculator" if "calculator" in required_tools else None
            })
            plan.append({
                "step": "perform_calculation",
                "description": "Perform the required calculation",
                "tool": "calculator" if "calculator" in required_tools else None
            })
            
        elif task_type == "search":
            plan.append({
                "step": "formulate_query",
                "description": "Create search query from task",
                "tool": "search" if "search" in required_tools else None
            })
            plan.append({
                "step": "execute_search",
                "description": "Search for information",
                "tool": "search" if "search" in required_tools else None
            })
            
        elif task_type == "analysis":
            plan.append({
                "step": "extract_content",
                "description": "Extract content to analyze",
                "tool": "text" if "text" in required_tools else None
            })
            plan.append({
                "step": "analyze_content",
                "description": "Analyze the content",
                "tool": "text" if "text" in required_tools else None
            })
            
        else:
            # Generic plan
            plan.append({
                "step": "process_task",
                "description": "Process the task using available tools",
                "tool": required_tools[0] if required_tools else None
            })
            
        logger.debug(f"Created plan with {len(plan)} steps")
        return plan
        
    def _execute_plan(self, plan: List[Dict[str, Any]], task: str) -> str:
        """Execute the plan using available tools."""
        results = []
        
        for step in plan:
            logger.debug(f"Executing step: {step['step']}")
            
            # Try to use specified tool
            if step["tool"]:
                tool = self._get_tool(step["tool"])
                if tool:
                    try:
                        result = tool.run(task)
                        results.append(f"{step['description']}: {result}")
                        continue
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        
            # Fallback to reasoning engine
            reasoning_result = self.reasoning_engine.process(
                f"{step['description']} for task: {task}"
            )
            results.append(f"{step['description']}: {reasoning_result}")
            
        # Compile results
        if results:
            return "\n".join(results)
        else:
            return "Task completed but no specific results generated."
            
    def _get_tool(self, tool_name: str) -> Optional[Any]:
        """Get a tool by name."""
        for tool in self.tools:
            if tool.name.lower() == tool_name.lower():
                return tool
        return None
        
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
        
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        state = super().get_state()
        state.update({
            "agent_type": "simple",
            "reasoning_type": "sequential",
            "tool_count": len(self.tools),
            "history_length": len(self.state_manager.get_history())
        })
        return state
        
    def reset(self):
        """Reset agent state."""
        super().reset()
        logger.info(f"SimpleAgent '{self.name}' reset complete")
