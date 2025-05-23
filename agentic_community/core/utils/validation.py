"""
Input validation utilities for the Agentic Framework.

This module provides validation functions and decorators to ensure
data integrity throughout the framework.
"""

import re
import functools
from typing import Any, Callable, Optional, List, Dict, Union, Type
from pydantic import BaseModel, ValidationError as PydanticValidationError

from agentic_community.core.exceptions import ValidationError


def validate_not_empty(value: str, field_name: str = "value") -> str:
    """
    Validate that a string is not empty or whitespace.
    
    Args:
        value: String to validate
        field_name: Name of the field for error messages
        
    Returns:
        The validated string (stripped)
        
    Raises:
        ValidationError: If string is empty or whitespace
    """
    if not value or not value.strip():
        raise ValidationError(field_name, value, "Cannot be empty")
    return value.strip()


def validate_string_length(
    value: str, 
    min_length: Optional[int] = None, 
    max_length: Optional[int] = None,
    field_name: str = "value"
) -> str:
    """
    Validate string length constraints.
    
    Args:
        value: String to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        field_name: Name of the field for error messages
        
    Returns:
        The validated string
        
    Raises:
        ValidationError: If string length is invalid
    """
    if min_length is not None and len(value) < min_length:
        raise ValidationError(
            field_name, 
            value, 
            f"Length must be at least {min_length} characters"
        )
    
    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            field_name, 
            value, 
            f"Length must not exceed {max_length} characters"
        )
    
    return value


def validate_regex(
    value: str, 
    pattern: str, 
    field_name: str = "value",
    error_message: Optional[str] = None
) -> str:
    """
    Validate string against a regex pattern.
    
    Args:
        value: String to validate
        pattern: Regex pattern
        field_name: Name of the field for error messages
        error_message: Custom error message
        
    Returns:
        The validated string
        
    Raises:
        ValidationError: If string doesn't match pattern
    """
    if not re.match(pattern, value):
        msg = error_message or f"Does not match required pattern: {pattern}"
        raise ValidationError(field_name, value, msg)
    
    return value


def validate_email(email: str) -> str:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        The validated email
        
    Raises:
        ValidationError: If email format is invalid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return validate_regex(
        email, 
        pattern, 
        "email",
        "Invalid email format"
    )


def validate_url(url: str) -> str:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        The validated URL
        
    Raises:
        ValidationError: If URL format is invalid
    """
    pattern = r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$'
    return validate_regex(
        url, 
        pattern, 
        "url",
        "Invalid URL format"
    )


def validate_integer_range(
    value: int,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    field_name: str = "value"
) -> int:
    """
    Validate integer is within range.
    
    Args:
        value: Integer to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        field_name: Name of the field for error messages
        
    Returns:
        The validated integer
        
    Raises:
        ValidationError: If integer is out of range
    """
    if min_value is not None and value < min_value:
        raise ValidationError(
            field_name, 
            value, 
            f"Must be at least {min_value}"
        )
    
    if max_value is not None and value > max_value:
        raise ValidationError(
            field_name, 
            value, 
            f"Must not exceed {max_value}"
        )
    
    return value


def validate_list_length(
    value: List[Any],
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    field_name: str = "list"
) -> List[Any]:
    """
    Validate list length constraints.
    
    Args:
        value: List to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        field_name: Name of the field for error messages
        
    Returns:
        The validated list
        
    Raises:
        ValidationError: If list length is invalid
    """
    if min_length is not None and len(value) < min_length:
        raise ValidationError(
            field_name, 
            f"List with {len(value)} items", 
            f"Must contain at least {min_length} items"
        )
    
    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            field_name, 
            f"List with {len(value)} items", 
            f"Must not exceed {max_length} items"
        )
    
    return value


def validate_type(
    value: Any,
    expected_type: Union[Type, tuple],
    field_name: str = "value"
) -> Any:
    """
    Validate value is of expected type.
    
    Args:
        value: Value to validate
        expected_type: Expected type or tuple of types
        field_name: Name of the field for error messages
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If value is not of expected type
    """
    if not isinstance(value, expected_type):
        type_name = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
        raise ValidationError(
            field_name,
            value,
            f"Expected type {type_name}, got {type(value).__name__}"
        )
    
    return value


def validate_pydantic_model(
    data: Dict[str, Any],
    model_class: Type[BaseModel],
    field_name: str = "data"
) -> BaseModel:
    """
    Validate data against a Pydantic model.
    
    Args:
        data: Dictionary data to validate
        model_class: Pydantic model class
        field_name: Name of the field for error messages
        
    Returns:
        Validated model instance
        
    Raises:
        ValidationError: If data doesn't match model
    """
    try:
        return model_class(**data)
    except PydanticValidationError as e:
        errors = []
        for error in e.errors():
            field = ".".join(str(loc) for loc in error['loc'])
            msg = error['msg']
            errors.append(f"{field}: {msg}")
        
        raise ValidationError(
            field_name,
            data,
            f"Model validation failed: {'; '.join(errors)}"
        )


def validate_agent_name(name: str) -> str:
    """
    Validate agent name.
    
    Args:
        name: Agent name to validate
        
    Returns:
        The validated name
        
    Raises:
        ValidationError: If name is invalid
    """
    name = validate_not_empty(name, "agent_name")
    name = validate_string_length(name, min_length=2, max_length=50, field_name="agent_name")
    # Allow alphanumeric, spaces, hyphens, and underscores
    name = validate_regex(
        name,
        r'^[a-zA-Z0-9\s\-_]+$',
        "agent_name",
        "Name can only contain letters, numbers, spaces, hyphens, and underscores"
    )
    return name


def validate_task(task: str) -> str:
    """
    Validate task input.
    
    Args:
        task: Task string to validate
        
    Returns:
        The validated task
        
    Raises:
        ValidationError: If task is invalid
    """
    task = validate_not_empty(task, "task")
    task = validate_string_length(task, min_length=5, max_length=5000, field_name="task")
    return task


# Validation decorators

def validate_inputs(**validators: Callable):
    """
    Decorator to validate function inputs.
    
    Usage:
        @validate_inputs(
            name=lambda x: validate_not_empty(x, "name"),
            age=lambda x: validate_integer_range(x, 0, 150, "age")
        )
        def create_person(name: str, age: int):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each argument
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    try:
                        bound_args.arguments[param_name] = validator(
                            bound_args.arguments[param_name]
                        )
                    except ValidationError:
                        raise
                    except Exception as e:
                        raise ValidationError(
                            param_name,
                            bound_args.arguments[param_name],
                            str(e)
                        )
            
            # Call function with validated arguments
            return func(*bound_args.args, **bound_args.kwargs)
        
        return wrapper
    return decorator


def validate_response(validator: Callable):
    """
    Decorator to validate function response.
    
    Usage:
        @validate_response(lambda x: validate_not_empty(x, "response"))
        def get_response() -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            try:
                return validator(result)
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(
                    "response",
                    result,
                    str(e)
                )
        
        return wrapper
    return decorator
