"""Tests for validation utilities."""

import pytest
from pydantic import BaseModel

from agentic_community.core.exceptions import ValidationError
from agentic_community.core.utils.validation import (
    validate_not_empty, validate_string_length, validate_regex,
    validate_email, validate_url, validate_integer_range,
    validate_list_length, validate_type, validate_pydantic_model,
    validate_agent_name, validate_task,
    validate_inputs, validate_response
)


class TestValidationFunctions:
    """Test validation utility functions."""
    
    def test_validate_not_empty(self):
        """Test validate_not_empty function."""
        # Valid cases
        assert validate_not_empty("hello") == "hello"
        assert validate_not_empty("  hello  ") == "hello"  # Strips whitespace
        
        # Invalid cases
        with pytest.raises(ValidationError) as exc_info:
            validate_not_empty("")
        assert "Cannot be empty" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            validate_not_empty("   ")
        assert "Cannot be empty" in str(exc_info.value)
    
    def test_validate_string_length(self):
        """Test validate_string_length function."""
        # Valid cases
        assert validate_string_length("hello", min_length=3, max_length=10) == "hello"
        assert validate_string_length("hi", min_length=2) == "hi"
        assert validate_string_length("hello", max_length=5) == "hello"
        
        # Invalid - too short
        with pytest.raises(ValidationError) as exc_info:
            validate_string_length("hi", min_length=3)
        assert "at least 3 characters" in str(exc_info.value)
        
        # Invalid - too long
        with pytest.raises(ValidationError) as exc_info:
            validate_string_length("hello world", max_length=5)
        assert "not exceed 5 characters" in str(exc_info.value)
    
    def test_validate_regex(self):
        """Test validate_regex function."""
        # Valid cases
        assert validate_regex("hello123", r"^[a-z0-9]+$") == "hello123"
        
        # Invalid cases
        with pytest.raises(ValidationError) as exc_info:
            validate_regex("Hello123", r"^[a-z0-9]+$")  # Uppercase not allowed
        assert "Does not match required pattern" in str(exc_info.value)
        
        # Custom error message
        with pytest.raises(ValidationError) as exc_info:
            validate_regex("test@", r"^[a-z]+$", error_message="Only letters allowed")
        assert "Only letters allowed" in str(exc_info.value)
    
    def test_validate_email(self):
        """Test validate_email function."""
        # Valid emails
        assert validate_email("test@example.com") == "test@example.com"
        assert validate_email("user.name+tag@example.co.uk") == "user.name+tag@example.co.uk"
        
        # Invalid emails
        invalid_emails = [
            "invalid",
            "@example.com",
            "test@",
            "test@.com",
            "test..test@example.com"
        ]
        
        for email in invalid_emails:
            with pytest.raises(ValidationError) as exc_info:
                validate_email(email)
            assert "Invalid email format" in str(exc_info.value)
    
    def test_validate_url(self):
        """Test validate_url function."""
        # Valid URLs
        assert validate_url("https://example.com") == "https://example.com"
        assert validate_url("http://sub.example.com/path?query=1") == "http://sub.example.com/path?query=1"
        
        # Invalid URLs
        invalid_urls = [
            "example.com",  # No protocol
            "ftp://example.com",  # Wrong protocol
            "https://",  # No domain
            "not a url"
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValidationError) as exc_info:
                validate_url(url)
            assert "Invalid URL format" in str(exc_info.value)
    
    def test_validate_integer_range(self):
        """Test validate_integer_range function."""
        # Valid cases
        assert validate_integer_range(5, min_value=1, max_value=10) == 5
        assert validate_integer_range(0, min_value=0) == 0
        assert validate_integer_range(100, max_value=100) == 100
        
        # Invalid - too small
        with pytest.raises(ValidationError) as exc_info:
            validate_integer_range(0, min_value=1)
        assert "Must be at least 1" in str(exc_info.value)
        
        # Invalid - too large
        with pytest.raises(ValidationError) as exc_info:
            validate_integer_range(11, max_value=10)
        assert "Must not exceed 10" in str(exc_info.value)
    
    def test_validate_list_length(self):
        """Test validate_list_length function."""
        # Valid cases
        assert validate_list_length([1, 2, 3], min_length=2, max_length=5) == [1, 2, 3]
        assert validate_list_length([], min_length=0) == []
        
        # Invalid - too short
        with pytest.raises(ValidationError) as exc_info:
            validate_list_length([1], min_length=2)
        assert "at least 2 items" in str(exc_info.value)
        
        # Invalid - too long
        with pytest.raises(ValidationError) as exc_info:
            validate_list_length([1, 2, 3, 4], max_length=3)
        assert "not exceed 3 items" in str(exc_info.value)
    
    def test_validate_type(self):
        """Test validate_type function."""
        # Valid cases
        assert validate_type("hello", str) == "hello"
        assert validate_type(42, int) == 42
        assert validate_type([1, 2], list) == [1, 2]
        assert validate_type(3.14, (int, float)) == 3.14
        
        # Invalid cases
        with pytest.raises(ValidationError) as exc_info:
            validate_type("hello", int)
        assert "Expected type int" in str(exc_info.value)
        assert "got str" in str(exc_info.value)
    
    def test_validate_pydantic_model(self):
        """Test validate_pydantic_model function."""
        
        class TestModel(BaseModel):
            name: str
            age: int
            email: str
        
        # Valid case
        valid_data = {"name": "John", "age": 30, "email": "john@example.com"}
        model = validate_pydantic_model(valid_data, TestModel)
        assert model.name == "John"
        assert model.age == 30
        
        # Invalid case - missing field
        with pytest.raises(ValidationError) as exc_info:
            validate_pydantic_model({"name": "John"}, TestModel)
        assert "Model validation failed" in str(exc_info.value)
        
        # Invalid case - wrong type
        with pytest.raises(ValidationError) as exc_info:
            validate_pydantic_model({"name": "John", "age": "thirty", "email": "john@example.com"}, TestModel)
        assert "Model validation failed" in str(exc_info.value)
    
    def test_validate_agent_name(self):
        """Test validate_agent_name function."""
        # Valid names
        assert validate_agent_name("MyAgent") == "MyAgent"
        assert validate_agent_name("Agent 123") == "Agent 123"
        assert validate_agent_name("Test-Agent_1") == "Test-Agent_1"
        
        # Invalid - empty
        with pytest.raises(ValidationError):
            validate_agent_name("")
        
        # Invalid - too short
        with pytest.raises(ValidationError):
            validate_agent_name("A")
        
        # Invalid - too long
        with pytest.raises(ValidationError):
            validate_agent_name("A" * 51)
        
        # Invalid - special characters
        with pytest.raises(ValidationError):
            validate_agent_name("Agent@123")
    
    def test_validate_task(self):
        """Test validate_task function."""
        # Valid tasks
        assert validate_task("Calculate 2 + 2") == "Calculate 2 + 2"
        assert validate_task("Help me plan a trip to Paris") == "Help me plan a trip to Paris"
        
        # Invalid - empty
        with pytest.raises(ValidationError):
            validate_task("")
        
        # Invalid - too short
        with pytest.raises(ValidationError):
            validate_task("Hi")
        
        # Invalid - too long
        with pytest.raises(ValidationError):
            validate_task("A" * 5001)


class TestValidationDecorators:
    """Test validation decorators."""
    
    def test_validate_inputs_decorator(self):
        """Test validate_inputs decorator."""
        
        @validate_inputs(
            name=lambda x: validate_not_empty(x, "name"),
            age=lambda x: validate_integer_range(x, 0, 150, "age")
        )
        def create_person(name: str, age: int) -> dict:
            return {"name": name, "age": age}
        
        # Valid inputs
        result = create_person("John", 30)
        assert result == {"name": "John", "age": 30}
        
        # Invalid name
        with pytest.raises(ValidationError):
            create_person("", 30)
        
        # Invalid age
        with pytest.raises(ValidationError):
            create_person("John", 200)
    
    def test_validate_response_decorator(self):
        """Test validate_response decorator."""
        
        @validate_response(lambda x: validate_not_empty(x, "response"))
        def get_message(valid: bool) -> str:
            return "Hello" if valid else ""
        
        # Valid response
        assert get_message(True) == "Hello"
        
        # Invalid response
        with pytest.raises(ValidationError):
            get_message(False)
