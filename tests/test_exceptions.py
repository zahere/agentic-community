"""Tests for exception handling."""

import pytest
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


class TestExceptions:
    """Test custom exception classes."""
    
    def test_base_exception(self):
        """Test AgenticError base exception."""
        error = AgenticError("Test error", error_code="TEST_ERROR", details={"key": "value"})
        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"key": "value"}
        
        # Test to_dict
        error_dict = error.to_dict()
        assert error_dict["error"] == "TEST_ERROR"
        assert error_dict["message"] == "Test error"
        assert error_dict["details"] == {"key": "value"}
    
    def test_api_key_error(self):
        """Test APIKeyError."""
        error = APIKeyError("TestProvider")
        assert "TestProvider" in str(error)
        assert error.error_code == "API_KEY_MISSING"
        assert error.details["provider"] == "TestProvider"
    
    def test_tool_not_found_error(self):
        """Test ToolNotFoundError."""
        error = ToolNotFoundError("calculator")
        assert "calculator" in str(error)
        assert error.error_code == "TOOL_NOT_FOUND"
        assert error.details["tool_name"] == "calculator"
    
    def test_tool_execution_error(self):
        """Test ToolExecutionError."""
        error = ToolExecutionError("calculator", "Division by zero")
        assert "calculator" in str(error)
        assert "Division by zero" in str(error)
        assert error.error_code == "TOOL_EXECUTION_FAILED"
    
    def test_tool_limit_exceeded(self):
        """Test ToolLimitExceeded."""
        error = ToolLimitExceeded(3, "community")
        assert "3" in str(error)
        assert "community" in str(error).lower()
        assert error.error_code == "TOOL_LIMIT_EXCEEDED"
    
    def test_agent_not_found_error(self):
        """Test AgentNotFoundError."""
        error = AgentNotFoundError("agent-123")
        assert "agent-123" in str(error)
        assert error.error_code == "AGENT_NOT_FOUND"
    
    def test_agent_execution_error(self):
        """Test AgentExecutionError."""
        error = AgentExecutionError("TestAgent", "Calculate 2+2", "LLM failure")
        assert "TestAgent" in str(error)
        assert "Calculate 2+2" in error.details["task"]
        assert "LLM failure" in str(error)
    
    def test_invalid_task_error(self):
        """Test InvalidTaskError."""
        error = InvalidTaskError("Task is empty")
        assert "Task is empty" in str(error)
        assert error.error_code == "INVALID_TASK"
    
    def test_state_not_found_error(self):
        """Test StateNotFoundError."""
        error = StateNotFoundError("agent-456")
        assert "agent-456" in str(error)
        assert error.error_code == "STATE_NOT_FOUND"
    
    def test_invalid_state_error(self):
        """Test InvalidStateError."""
        error = InvalidStateError("Corrupted state data")
        assert "Corrupted state data" in str(error)
        assert error.error_code == "INVALID_STATE"
    
    def test_feature_not_available_error(self):
        """Test FeatureNotAvailableError."""
        error = FeatureNotAvailableError("multi-agent", "community", "enterprise")
        assert "multi-agent" in str(error)
        assert "community" in str(error)
        assert "enterprise" in str(error)
        assert error.error_code == "FEATURE_NOT_AVAILABLE"
    
    def test_search_error(self):
        """Test SearchError."""
        error = SearchError("python tutorial", "Connection timeout")
        assert "python tutorial" in str(error)
        assert "Connection timeout" in str(error)
        assert error.error_code == "SEARCH_FAILED"
    
    def test_api_error(self):
        """Test APIError."""
        error = APIError("/api/agents", 500, "Internal server error")
        assert "/api/agents" in str(error)
        assert "500" in str(error)
        assert "Internal server error" in str(error)
        assert error.details["status_code"] == 500
    
    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("email", "invalid@", "Invalid email format")
        assert "email" in str(error)
        assert "Invalid email format" in str(error)
        assert error.details["value"] == "invalid@"
    
    def test_timeout_error(self):
        """Test TimeoutError."""
        error = TimeoutError("search", 30.0)
        assert "search" in str(error)
        assert "30" in str(error)
        assert error.details["timeout"] == 30.0
    
    def test_handle_error_with_agentic_error(self, caplog):
        """Test handle_error with AgenticError."""
        error = ToolNotFoundError("calculator")
        handle_error(error, "tool lookup")
        
        assert "tool lookup failed" in caplog.text
        assert "TOOL_NOT_FOUND" in caplog.text
    
    def test_handle_error_with_generic_exception(self, caplog):
        """Test handle_error with generic exception."""
        error = ValueError("Invalid value")
        handle_error(error, "value processing")
        
        assert "Unexpected error during value processing" in caplog.text
