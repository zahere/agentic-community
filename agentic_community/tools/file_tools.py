"""
File handling tool implementation for Agentic Framework.

This module provides file reading and writing functionality.
"""

import os
import json
import csv
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

from agentic_community.core.base import BaseTool
from agentic_community.core.exceptions import (
    ToolExecutionError, ValidationError, handle_error
)
from agentic_community.core.utils.validation import validate_not_empty

logger = logging.getLogger(__name__)


class FileReaderTool(BaseTool):
    """
    Tool for reading files in various formats.
    
    Supports text, JSON, and CSV files.
    """
    
    name = "file_reader"
    description = "Read content from files (text, JSON, CSV)"
    
    def __init__(self, base_path: Optional[str] = None, max_file_size: int = 10 * 1024 * 1024):
        """
        Initialize file reader tool.
        
        Args:
            base_path: Base directory for file operations (for security)
            max_file_size: Maximum file size in bytes (default: 10MB)
        """
        super().__init__()
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.max_file_size = max_file_size
    
    def _validate_path(self, file_path: str) -> Path:
        """
        Validate and resolve file path.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Resolved Path object
            
        Raises:
            ValidationError: If path is invalid or outside base_path
        """
        try:
            path = Path(file_path)
            
            # If relative, make it relative to base_path
            if not path.is_absolute():
                path = self.base_path / path
            
            # Resolve to absolute path
            resolved_path = path.resolve()
            
            # Check if path is within base_path (security)
            if self.base_path not in resolved_path.parents and resolved_path != self.base_path:
                raise ValidationError(
                    "file_path",
                    str(file_path),
                    f"Path must be within {self.base_path}"
                )
            
            # Check if file exists
            if not resolved_path.exists():
                raise ValidationError(
                    "file_path",
                    str(file_path),
                    "File does not exist"
                )
            
            # Check if it's a file
            if not resolved_path.is_file():
                raise ValidationError(
                    "file_path",
                    str(file_path),
                    "Path is not a file"
                )
            
            # Check file size
            if resolved_path.stat().st_size > self.max_file_size:
                raise ValidationError(
                    "file_path",
                    str(file_path),
                    f"File size exceeds maximum of {self.max_file_size} bytes"
                )
            
            return resolved_path
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError("file_path", str(file_path), str(e))
    
    def read_text(self, file_path: str, encoding: str = "utf-8") -> str:
        """
        Read text file content.
        
        Args:
            file_path: Path to the text file
            encoding: File encoding (default: utf-8)
            
        Returns:
            File content as string
        """
        path = self._validate_path(file_path)
        
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            raise ToolExecutionError(self.name, f"Failed to read text file: {str(e)}")
    
    def read_json(self, file_path: str) -> Dict[str, Any]:
        """
        Read JSON file content.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Parsed JSON content
        """
        path = self._validate_path(file_path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ToolExecutionError(self.name, f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ToolExecutionError(self.name, f"Failed to read JSON file: {str(e)}")
    
    def read_csv(self, file_path: str, delimiter: str = ',', has_header: bool = True) -> List[Dict[str, str]]:
        """
        Read CSV file content.
        
        Args:
            file_path: Path to the CSV file
            delimiter: CSV delimiter (default: comma)
            has_header: Whether the CSV has a header row
            
        Returns:
            List of dictionaries representing CSV rows
        """
        path = self._validate_path(file_path)
        
        try:
            rows = []
            with open(path, 'r', encoding='utf-8', newline='') as f:
                if has_header:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    rows = list(reader)
                else:
                    reader = csv.reader(f, delimiter=delimiter)
                    rows = [{"col_" + str(i): value for i, value in enumerate(row)} 
                           for row in reader]
            return rows
        except Exception as e:
            raise ToolExecutionError(self.name, f"Failed to read CSV file: {str(e)}")
    
    def _invoke(self, file_path: str) -> str:
        """
        Tool invocation method used by agents.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            File content as string
        """
        validate_not_empty(file_path, "file_path")
        
        # Determine file type by extension
        path = Path(file_path)
        extension = path.suffix.lower()
        
        try:
            if extension == '.json':
                content = self.read_json(file_path)
                return json.dumps(content, indent=2)
            elif extension == '.csv':
                rows = self.read_csv(file_path)
                return f"CSV with {len(rows)} rows:\n" + json.dumps(rows[:5], indent=2) + "\n..."
            else:
                # Default to text
                return self.read_text(file_path)
        except Exception as e:
            handle_error(e, f"reading file {file_path}")
            return f"Error reading file: {str(e)}"


class FileWriterTool(BaseTool):
    """
    Tool for writing files in various formats.
    
    Supports text, JSON, and CSV files.
    """
    
    name = "file_writer"
    description = "Write content to files (text, JSON, CSV)"
    
    def __init__(self, base_path: Optional[str] = None, max_file_size: int = 10 * 1024 * 1024):
        """
        Initialize file writer tool.
        
        Args:
            base_path: Base directory for file operations (for security)
            max_file_size: Maximum file size in bytes (default: 10MB)
        """
        super().__init__()
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.max_file_size = max_file_size
    
    def _validate_write_path(self, file_path: str) -> Path:
        """
        Validate and prepare path for writing.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Resolved Path object
            
        Raises:
            ValidationError: If path is invalid
        """
        try:
            path = Path(file_path)
            
            # If relative, make it relative to base_path
            if not path.is_absolute():
                path = self.base_path / path
            
            # Resolve to absolute path
            resolved_path = path.resolve()
            
            # Check if path is within base_path (security)
            if self.base_path not in resolved_path.parents and resolved_path != self.base_path:
                raise ValidationError(
                    "file_path",
                    str(file_path),
                    f"Path must be within {self.base_path}"
                )
            
            # Create parent directories if they don't exist
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            
            return resolved_path
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError("file_path", str(file_path), str(e))
    
    def write_text(self, file_path: str, content: str, encoding: str = "utf-8", append: bool = False) -> str:
        """
        Write text to file.
        
        Args:
            file_path: Path to the file
            content: Content to write
            encoding: File encoding (default: utf-8)
            append: Whether to append to existing file
            
        Returns:
            Success message
        """
        path = self._validate_write_path(file_path)
        
        # Check content size
        if len(content.encode(encoding)) > self.max_file_size:
            raise ValidationError("content", "too large", f"Content exceeds maximum size of {self.max_file_size} bytes")
        
        try:
            mode = 'a' if append else 'w'
            with open(path, mode, encoding=encoding) as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            raise ToolExecutionError(self.name, f"Failed to write text file: {str(e)}")
    
    def write_json(self, file_path: str, data: Union[Dict, List], indent: int = 2) -> str:
        """
        Write JSON data to file.
        
        Args:
            file_path: Path to the file
            data: Data to write (dict or list)
            indent: JSON indentation
            
        Returns:
            Success message
        """
        path = self._validate_write_path(file_path)
        
        try:
            json_content = json.dumps(data, indent=indent)
            
            # Check content size
            if len(json_content.encode('utf-8')) > self.max_file_size:
                raise ValidationError("data", "too large", f"JSON content exceeds maximum size of {self.max_file_size} bytes")
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json_content)
            return f"Successfully wrote JSON to {path}"
        except json.JSONEncodeError as e:
            raise ToolExecutionError(self.name, f"Failed to encode JSON: {str(e)}")
        except Exception as e:
            if isinstance(e, (ValidationError, ToolExecutionError)):
                raise
            raise ToolExecutionError(self.name, f"Failed to write JSON file: {str(e)}")
    
    def write_csv(self, file_path: str, data: List[Dict[str, Any]], delimiter: str = ',') -> str:
        """
        Write CSV data to file.
        
        Args:
            file_path: Path to the file
            data: List of dictionaries to write
            delimiter: CSV delimiter
            
        Returns:
            Success message
        """
        path = self._validate_write_path(file_path)
        
        if not data:
            raise ValidationError("data", data, "Cannot write empty CSV")
        
        try:
            # Get all unique keys for headers
            headers = set()
            for row in data:
                headers.update(row.keys())
            headers = sorted(list(headers))
            
            with open(path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers, delimiter=delimiter)
                writer.writeheader()
                writer.writerows(data)
            
            return f"Successfully wrote CSV with {len(data)} rows to {path}"
        except Exception as e:
            raise ToolExecutionError(self.name, f"Failed to write CSV file: {str(e)}")
    
    def _invoke(self, params: str) -> str:
        """
        Tool invocation method used by agents.
        
        Expected format: "filepath: content" or JSON with filepath and content
        
        Args:
            params: Parameters containing filepath and content
            
        Returns:
            Success message
        """
        try:
            # Try to parse as JSON first
            try:
                params_dict = json.loads(params)
                file_path = params_dict.get('filepath', '')
                content = params_dict.get('content', '')
                file_type = params_dict.get('type', 'text')
            except:
                # Fallback to simple format
                if ':' in params:
                    file_path, content = params.split(':', 1)
                    file_path = file_path.strip()
                    content = content.strip()
                    file_type = 'text'
                else:
                    raise ValidationError("params", params, "Invalid format. Use 'filepath: content' or JSON")
            
            validate_not_empty(file_path, "file_path")
            validate_not_empty(content, "content")
            
            # Determine file type
            if file_type == 'json' or file_path.endswith('.json'):
                data = json.loads(content) if isinstance(content, str) else content
                return self.write_json(file_path, data)
            elif file_type == 'csv' or file_path.endswith('.csv'):
                data = json.loads(content) if isinstance(content, str) else content
                return self.write_csv(file_path, data)
            else:
                return self.write_text(file_path, content)
                
        except Exception as e:
            handle_error(e, f"writing file")
            return f"Error writing file: {str(e)}"
