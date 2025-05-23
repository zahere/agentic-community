"""
Core module for Agentic Framework.
"""

# Import all base classes and utilities
from agentic_community.core.base import BaseAgent, BaseTool, ToolConfig
from agentic_community.core.reasoning import ReasoningEngine
from agentic_community.core.state import StateManager

# Import exceptions
from agentic_community.core.exceptions import (
    AgenticError, ConfigurationError, APIKeyError,
    ToolError, ToolNotFoundError, ToolExecutionError, ToolLimitExceeded,
    AgentError, AgentNotFoundError, AgentExecutionError, InvalidTaskError,
    StateError, StateNotFoundError, InvalidStateError,
    LicenseError, FeatureNotAvailableError,
    NetworkError, SearchError, APIError,
    ValidationError, TimeoutError,
    handle_error
)

# Import validation utilities
from agentic_community.core.utils.validation import (
    validate_not_empty, validate_string_length, validate_regex,
    validate_email, validate_url, validate_integer_range,
    validate_list_length, validate_type, validate_pydantic_model,
    validate_agent_name, validate_task,
    validate_inputs, validate_response
)

__all__ = [
    # Base classes
    "BaseAgent",
    "BaseTool",
    "ToolConfig",
    "ReasoningEngine",
    "StateManager",
    
    # Exceptions
    "AgenticError",
    "ConfigurationError",
    "APIKeyError",
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ToolLimitExceeded",
    "AgentError",
    "AgentNotFoundError",
    "AgentExecutionError",
    "InvalidTaskError",
    "StateError",
    "StateNotFoundError",
    "InvalidStateError",
    "LicenseError",
    "FeatureNotAvailableError",
    "NetworkError",
    "SearchError",
    "APIError",
    "ValidationError",
    "TimeoutError",
    "handle_error",
    
    # Validation
    "validate_not_empty",
    "validate_string_length",
    "validate_regex",
    "validate_email",
    "validate_url",
    "validate_integer_range",
    "validate_list_length",
    "validate_type",
    "validate_pydantic_model",
    "validate_agent_name",
    "validate_task",
    "validate_inputs",
    "validate_response",
]
