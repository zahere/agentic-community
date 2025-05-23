"""
Custom exceptions for the Agentic Framework.

This module defines all custom exceptions used throughout the framework,
providing clear error messages and proper exception hierarchy.
"""

from typing import Optional, Any, Dict


class AgenticError(Exception):
    """Base exception for all Agentic Framework errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ConfigurationError(AgenticError):
    """Raised when there's a configuration problem."""
    pass


class APIKeyError(ConfigurationError):
    """Raised when API key is missing or invalid."""
    
    def __init__(self, provider: str = "OpenAI"):
        super().__init__(
            f"{provider} API key not found. Please set the appropriate environment variable.",
            error_code="API_KEY_MISSING",
            details={"provider": provider}
        )


class ToolError(AgenticError):
    """Base exception for tool-related errors."""
    pass


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found."""
    
    def __init__(self, tool_name: str):
        super().__init__(
            f"Tool '{tool_name}' not found",
            error_code="TOOL_NOT_FOUND",
            details={"tool_name": tool_name}
        )


class ToolExecutionError(ToolError):
    """Raised when a tool fails during execution."""
    
    def __init__(self, tool_name: str, error: str):
        super().__init__(
            f"Tool '{tool_name}' failed: {error}",
            error_code="TOOL_EXECUTION_FAILED",
            details={"tool_name": tool_name, "error": error}
        )


class ToolLimitExceeded(ToolError):
    """Raised when trying to add more tools than allowed."""
    
    def __init__(self, limit: int, edition: str = "community"):
        super().__init__(
            f"Tool limit exceeded. {edition.capitalize()} edition supports up to {limit} tools.",
            error_code="TOOL_LIMIT_EXCEEDED",
            details={"limit": limit, "edition": edition}
        )


class AgentError(AgenticError):
    """Base exception for agent-related errors."""
    pass


class AgentNotFoundError(AgentError):
    """Raised when a requested agent is not found."""
    
    def __init__(self, agent_id: str):
        super().__init__(
            f"Agent with ID '{agent_id}' not found",
            error_code="AGENT_NOT_FOUND",
            details={"agent_id": agent_id}
        )


class AgentExecutionError(AgentError):
    """Raised when an agent fails during task execution."""
    
    def __init__(self, agent_name: str, task: str, error: str):
        super().__init__(
            f"Agent '{agent_name}' failed to execute task: {error}",
            error_code="AGENT_EXECUTION_FAILED",
            details={"agent_name": agent_name, "task": task, "error": error}
        )


class InvalidTaskError(AgentError):
    """Raised when task input is invalid."""
    
    def __init__(self, reason: str):
        super().__init__(
            f"Invalid task: {reason}",
            error_code="INVALID_TASK",
            details={"reason": reason}
        )


class StateError(AgenticError):
    """Base exception for state-related errors."""
    pass


class StateNotFoundError(StateError):
    """Raised when requested state is not found."""
    
    def __init__(self, agent_id: str):
        super().__init__(
            f"State for agent '{agent_id}' not found",
            error_code="STATE_NOT_FOUND",
            details={"agent_id": agent_id}
        )


class InvalidStateError(StateError):
    """Raised when state data is invalid or corrupted."""
    
    def __init__(self, reason: str):
        super().__init__(
            f"Invalid state: {reason}",
            error_code="INVALID_STATE",
            details={"reason": reason}
        )


class LicenseError(AgenticError):
    """Base exception for licensing errors."""
    pass


class FeatureNotAvailableError(LicenseError):
    """Raised when trying to use a feature not available in current edition."""
    
    def __init__(self, feature: str, edition: str = "community", required_edition: str = "enterprise"):
        super().__init__(
            f"Feature '{feature}' is not available in {edition} edition. Upgrade to {required_edition} edition.",
            error_code="FEATURE_NOT_AVAILABLE",
            details={
                "feature": feature,
                "current_edition": edition,
                "required_edition": required_edition
            }
        )


class NetworkError(AgenticError):
    """Base exception for network-related errors."""
    pass


class SearchError(NetworkError):
    """Raised when search operation fails."""
    
    def __init__(self, query: str, error: str):
        super().__init__(
            f"Search failed for query '{query}': {error}",
            error_code="SEARCH_FAILED",
            details={"query": query, "error": error}
        )


class APIError(NetworkError):
    """Raised when API request fails."""
    
    def __init__(self, endpoint: str, status_code: int, error: str):
        super().__init__(
            f"API request to '{endpoint}' failed with status {status_code}: {error}",
            error_code="API_REQUEST_FAILED",
            details={
                "endpoint": endpoint,
                "status_code": status_code,
                "error": error
            }
        )


class ValidationError(AgenticError):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"Validation failed for '{field}': {reason}",
            error_code="VALIDATION_FAILED",
            details={
                "field": field,
                "value": str(value),
                "reason": reason
            }
        )


class TimeoutError(AgenticError):
    """Raised when an operation times out."""
    
    def __init__(self, operation: str, timeout: float):
        super().__init__(
            f"Operation '{operation}' timed out after {timeout} seconds",
            error_code="OPERATION_TIMEOUT",
            details={
                "operation": operation,
                "timeout": timeout
            }
        )


# Utility function for handling errors
def handle_error(error: Exception, operation: str = "operation") -> None:
    """
    Handle an error appropriately based on its type.
    
    Args:
        error: The exception to handle
        operation: Description of the operation that failed
    """
    if isinstance(error, AgenticError):
        # Log structured error information
        import logging
        logger = logging.getLogger(__name__)
        logger.error(
            f"{operation} failed",
            extra={
                "error_code": error.error_code,
                "details": error.details
            }
        )
    else:
        # Log unexpected errors
        import logging
        logger = logging.getLogger(__name__)
        logger.exception(f"Unexpected error during {operation}")
