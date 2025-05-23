"""
Basic Reasoning Module - Community Edition
Simple sequential reasoning capabilities
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field

from ..utils import get_logger

logger = get_logger(__name__)


class ThoughtStep(BaseModel):
    """Represents a single thought step."""
    step_number: int = Field(description="Step number in sequence")
    thought: str = Field(description="The thought content")
    action: Optional[str] = Field(default=None, description="Associated action")
    confidence: float = Field(default=0.8, description="Confidence level")


class BasicReasoner:
    """
    Basic reasoning engine for community edition.
    Supports only sequential thinking without reflection.
    """
    
    def __init__(self, max_steps: int = 5):
        """
        Initialize basic reasoner.
        
        Args:
            max_steps: Maximum reasoning steps (community limited to 5)
        """
        self.max_steps = min(max_steps, 5)  # Community limit
        logger.info(f"Initialized BasicReasoner with max_steps={self.max_steps}")
        
    def decompose_task(self, task: str) -> List[ThoughtStep]:
        """
        Decompose a task into simple sequential steps.
        
        Args:
            task: The task to decompose
            
        Returns:
            List of thought steps
        """
        logger.debug(f"Decomposing task: {task}")
        
        # Simple keyword-based decomposition
        steps = []
        
        # Identify task type
        task_lower = task.lower()
        
        if "plan" in task_lower:
            steps = self._decompose_planning_task(task)
        elif "calculate" in task_lower or "compute" in task_lower:
            steps = self._decompose_calculation_task(task)
        elif "analyze" in task_lower or "summarize" in task_lower:
            steps = self._decompose_analysis_task(task)
        else:
            steps = self._decompose_generic_task(task)
            
        # Limit to max steps
        return steps[:self.max_steps]
        
    def _decompose_planning_task(self, task: str) -> List[ThoughtStep]:
        """Decompose a planning task."""
        return [
            ThoughtStep(
                step_number=1,
                thought="Identify the main objective",
                action="extract_goal"
            ),
            ThoughtStep(
                step_number=2,
                thought="List required resources",
                action="identify_resources"
            ),
            ThoughtStep(
                step_number=3,
                thought="Create timeline",
                action="schedule_tasks"
            ),
            ThoughtStep(
                step_number=4,
                thought="Identify potential challenges",
                action="risk_assessment"
            ),
            ThoughtStep(
                step_number=5,
                thought="Finalize the plan",
                action="compile_results"
            )
        ]
        
    def _decompose_calculation_task(self, task: str) -> List[ThoughtStep]:
        """Decompose a calculation task."""
        return [
            ThoughtStep(
                step_number=1,
                thought="Identify numbers and operations",
                action="parse_expression"
            ),
            ThoughtStep(
                step_number=2,
                thought="Perform calculation",
                action="calculate"
            ),
            ThoughtStep(
                step_number=3,
                thought="Verify result",
                action="validate"
            )
        ]
        
    def _decompose_analysis_task(self, task: str) -> List[ThoughtStep]:
        """Decompose an analysis task."""
        return [
            ThoughtStep(
                step_number=1,
                thought="Gather relevant information",
                action="collect_data"
            ),
            ThoughtStep(
                step_number=2,
                thought="Identify key points",
                action="extract_insights"
            ),
            ThoughtStep(
                step_number=3,
                thought="Organize findings",
                action="structure_results"
            ),
            ThoughtStep(
                step_number=4,
                thought="Create summary",
                action="summarize"
            )
        ]
        
    def _decompose_generic_task(self, task: str) -> List[ThoughtStep]:
        """Decompose a generic task."""
        return [
            ThoughtStep(
                step_number=1,
                thought="Understand the task requirements",
                action="analyze_request"
            ),
            ThoughtStep(
                step_number=2,
                thought="Break down into subtasks",
                action="decompose"
            ),
            ThoughtStep(
                step_number=3,
                thought="Execute each subtask",
                action="execute"
            ),
            ThoughtStep(
                step_number=4,
                thought="Combine results",
                action="integrate"
            ),
            ThoughtStep(
                step_number=5,
                thought="Present final output",
                action="format_response"
            )
        ]
        
    def reason(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform basic reasoning on a task.
        
        Args:
            task: The task to reason about
            context: Optional context
            
        Returns:
            Reasoning results
        """
        # Decompose task
        steps = self.decompose_task(task)
        
        # Execute reasoning (simplified for community edition)
        results = {
            "task": task,
            "reasoning_type": "sequential",
            "steps": [step.model_dump() for step in steps],
            "confidence": sum(s.confidence for s in steps) / len(steps),
            "limitations": [
                "No self-reflection capability",
                "Limited to sequential reasoning",
                "No multi-path exploration"
            ]
        }
        
        return results
