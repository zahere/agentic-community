"""
Workflow Automation for Agentic Community Edition

This module provides workflow automation capabilities for orchestrating
complex multi-step agent tasks with dependencies, conditions, and branching.
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging

from ..agents.base import BaseAgent
from ..core.exceptions import WorkflowError


class TaskStatus(Enum):
    """Status of a workflow task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Type of workflow task."""
    AGENT = "agent"
    CONDITION = "condition"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    LOOP = "loop"
    HUMAN_INPUT = "human_input"


@dataclass
class TaskResult:
    """Result of a workflow task execution."""
    task_id: str
    status: TaskStatus
    output: Any
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowTask:
    """Definition of a workflow task."""
    id: str
    name: str
    type: TaskType
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    retry_policy: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


class WorkflowEngine:
    """Engine for executing workflows with agents."""
    
    def __init__(self, name: str = "WorkflowEngine"):
        self.name = name
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, List[WorkflowTask]] = {}
        self.results: Dict[str, Dict[str, TaskResult]] = {}
        self.logger = logging.getLogger(__name__)
        
    def register_agent(self, agent_id: str, agent: BaseAgent):
        """Register an agent for use in workflows."""
        self.agents[agent_id] = agent
        self.logger.info(f"Registered agent: {agent_id}")
        
    def create_workflow(self, workflow_id: str, tasks: List[WorkflowTask]):
        """Create a new workflow definition."""
        self.workflows[workflow_id] = tasks
        self.results[workflow_id] = {}
        self.logger.info(f"Created workflow: {workflow_id} with {len(tasks)} tasks")
        
    async def execute_workflow(
        self,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, TaskResult]:
        """Execute a workflow and return results."""
        if workflow_id not in self.workflows:
            raise WorkflowError(f"Workflow {workflow_id} not found")
            
        tasks = self.workflows[workflow_id]
        context = context or {}
        
        self.logger.info(f"Starting workflow execution: {workflow_id}")
        
        # Build task graph
        task_map = {task.id: task for task in tasks}
        
        # Execute tasks in topological order
        completed_tasks = set()
        
        while len(completed_tasks) < len(tasks):
            # Find tasks ready to execute
            ready_tasks = []
            for task in tasks:
                if task.id not in completed_tasks:
                    # Check dependencies
                    deps_satisfied = all(
                        dep_id in completed_tasks 
                        for dep_id in task.dependencies
                    )
                    if deps_satisfied:
                        # Check conditions
                        if await self._evaluate_conditions(task, context):
                            ready_tasks.append(task)
                        else:
                            # Skip task if conditions not met
                            self.results[workflow_id][task.id] = TaskResult(
                                task_id=task.id,
                                status=TaskStatus.SKIPPED,
                                output=None
                            )
                            completed_tasks.add(task.id)
            
            if not ready_tasks and len(completed_tasks) < len(tasks):
                raise WorkflowError("Circular dependency detected in workflow")
            
            # Execute ready tasks
            if ready_tasks:
                if len(ready_tasks) == 1:
                    # Sequential execution
                    task = ready_tasks[0]
                    result = await self._execute_task(task, context)
                    self.results[workflow_id][task.id] = result
                    completed_tasks.add(task.id)
                    
                    # Update context with task output
                    if result.status == TaskStatus.COMPLETED:
                        context[f"task_{task.id}_output"] = result.output
                else:
                    # Parallel execution
                    tasks_to_run = ready_tasks[:5]  # Limit parallelism
                    results = await asyncio.gather(
                        *[self._execute_task(task, context) for task in tasks_to_run],
                        return_exceptions=True
                    )
                    
                    for task, result in zip(tasks_to_run, results):
                        if isinstance(result, Exception):
                            result = TaskResult(
                                task_id=task.id,
                                status=TaskStatus.FAILED,
                                output=None,
                                error=str(result)
                            )
                        self.results[workflow_id][task.id] = result
                        completed_tasks.add(task.id)
                        
                        if result.status == TaskStatus.COMPLETED:
                            context[f"task_{task.id}_output"] = result.output
        
        self.logger.info(f"Workflow execution completed: {workflow_id}")
        return self.results[workflow_id]
        
    async def _execute_task(
        self,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> TaskResult:
        """Execute a single task."""
        self.logger.info(f"Executing task: {task.name} ({task.type.value})")
        
        result = TaskResult(
            task_id=task.id,
            status=TaskStatus.RUNNING,
            output=None,
            started_at=datetime.now()
        )
        
        try:
            if task.type == TaskType.AGENT:
                output = await self._execute_agent_task(task, context)
            elif task.type == TaskType.CONDITION:
                output = await self._execute_condition_task(task, context)
            elif task.type == TaskType.PARALLEL:
                output = await self._execute_parallel_task(task, context)
            elif task.type == TaskType.SEQUENTIAL:
                output = await self._execute_sequential_task(task, context)
            elif task.type == TaskType.LOOP:
                output = await self._execute_loop_task(task, context)
            elif task.type == TaskType.HUMAN_INPUT:
                output = await self._execute_human_input_task(task, context)
            else:
                raise WorkflowError(f"Unknown task type: {task.type}")
                
            result.status = TaskStatus.COMPLETED
            result.output = output
            
        except Exception as e:
            self.logger.error(f"Task failed: {task.name} - {str(e)}")
            result.status = TaskStatus.FAILED
            result.error = str(e)
            
            # Handle retry policy
            if task.retry_policy:
                retries = task.retry_policy.get("max_retries", 0)
                if retries > 0:
                    # Implement retry logic here
                    pass
                    
        result.completed_at = datetime.now()
        return result
        
    async def _execute_agent_task(
        self,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> Any:
        """Execute a task using an agent."""
        agent_id = task.config.get("agent_id")
        if agent_id not in self.agents:
            raise WorkflowError(f"Agent {agent_id} not found")
            
        agent = self.agents[agent_id]
        prompt = task.config.get("prompt", "")
        
        # Substitute context variables in prompt
        for key, value in context.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
            
        # Execute agent
        if hasattr(agent, 'arun'):
            response = await agent.arun(prompt)
        else:
            response = await asyncio.to_thread(agent.run, prompt)
            
        return response
        
    async def _execute_condition_task(
        self,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> bool:
        """Execute a conditional task."""
        condition = task.config.get("condition", {})
        return await self._evaluate_condition(condition, context)
        
    async def _execute_parallel_task(
        self,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> List[Any]:
        """Execute subtasks in parallel."""
        subtasks = task.config.get("tasks", [])
        results = await asyncio.gather(
            *[self._execute_task(subtask, context) for subtask in subtasks]
        )
        return [r.output for r in results if r.status == TaskStatus.COMPLETED]
        
    async def _execute_sequential_task(
        self,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> Any:
        """Execute subtasks sequentially."""
        subtasks = task.config.get("tasks", [])
        last_output = None
        
        for subtask in subtasks:
            result = await self._execute_task(subtask, context)
            if result.status == TaskStatus.COMPLETED:
                last_output = result.output
                context[f"previous_output"] = last_output
            else:
                break
                
        return last_output
        
    async def _execute_loop_task(
        self,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> List[Any]:
        """Execute a loop task."""
        items = task.config.get("items", [])
        loop_task = task.config.get("task")
        outputs = []
        
        for i, item in enumerate(items):
            loop_context = context.copy()
            loop_context["loop_index"] = i
            loop_context["loop_item"] = item
            
            result = await self._execute_task(loop_task, loop_context)
            if result.status == TaskStatus.COMPLETED:
                outputs.append(result.output)
                
        return outputs
        
    async def _execute_human_input_task(
        self,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> str:
        """Execute a human input task."""
        prompt = task.config.get("prompt", "Please provide input:")
        # In a real implementation, this would integrate with a UI
        # For now, we'll return a placeholder
        return f"[Human input required: {prompt}]"
        
    async def _evaluate_conditions(
        self,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate all conditions for a task."""
        if not task.conditions:
            return True
            
        for condition in task.conditions:
            if not await self._evaluate_condition(condition, context):
                return False
                
        return True
        
    async def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate a single condition."""
        cond_type = condition.get("type", "equals")
        field = condition.get("field")
        value = condition.get("value")
        
        if field not in context:
            return False
            
        context_value = context[field]
        
        if cond_type == "equals":
            return context_value == value
        elif cond_type == "not_equals":
            return context_value != value
        elif cond_type == "contains":
            return value in str(context_value)
        elif cond_type == "greater_than":
            return float(context_value) > float(value)
        elif cond_type == "less_than":
            return float(context_value) < float(value)
        elif cond_type == "in":
            return context_value in value
        elif cond_type == "not_in":
            return context_value not in value
        else:
            return False
            
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow."""
        if workflow_id not in self.results:
            return {"status": "not_found"}
            
        results = self.results[workflow_id]
        
        total_tasks = len(self.workflows[workflow_id])
        completed_tasks = sum(
            1 for r in results.values() 
            if r.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]
        )
        failed_tasks = sum(
            1 for r in results.values() 
            if r.status == TaskStatus.FAILED
        )
        
        if failed_tasks > 0:
            status = "failed"
        elif completed_tasks == total_tasks:
            status = "completed"
        else:
            status = "running"
            
        return {
            "status": status,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "results": results
        }
        
    def save_workflow(self, workflow_id: str, filepath: str):
        """Save workflow definition to file."""
        if workflow_id not in self.workflows:
            raise WorkflowError(f"Workflow {workflow_id} not found")
            
        workflow_data = {
            "id": workflow_id,
            "tasks": [
                {
                    "id": task.id,
                    "name": task.name,
                    "type": task.type.value,
                    "config": task.config,
                    "dependencies": task.dependencies,
                    "conditions": task.conditions,
                    "retry_policy": task.retry_policy,
                    "timeout": task.timeout
                }
                for task in self.workflows[workflow_id]
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(workflow_data, f, indent=2)
            
    def load_workflow(self, filepath: str) -> str:
        """Load workflow definition from file."""
        with open(filepath, 'r') as f:
            workflow_data = json.load(f)
            
        workflow_id = workflow_data["id"]
        tasks = []
        
        for task_data in workflow_data["tasks"]:
            task = WorkflowTask(
                id=task_data["id"],
                name=task_data["name"],
                type=TaskType(task_data["type"]),
                config=task_data["config"],
                dependencies=task_data.get("dependencies", []),
                conditions=task_data.get("conditions", []),
                retry_policy=task_data.get("retry_policy"),
                timeout=task_data.get("timeout")
            )
            tasks.append(task)
            
        self.create_workflow(workflow_id, tasks)
        return workflow_id
