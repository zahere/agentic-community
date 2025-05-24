"""
Workflow automation module for Agentic Community Edition.
"""

from .workflow_engine import (
    WorkflowEngine,
    WorkflowTask,
    TaskType,
    TaskStatus,
    TaskResult
)

__all__ = [
    "WorkflowEngine",
    "WorkflowTask",
    "TaskType",
    "TaskStatus",
    "TaskResult"
]
