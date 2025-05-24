"""
File Tool for Community Edition

Provides file operations including reading, writing, and processing
various file formats.
"""

import os
import json
import csv
import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import aiofiles
import yaml
import mimetypes
from datetime import datetime

from agentic_community.core.base import BaseTool
from agentic_community.core.exceptions import ToolError


class FileTool(BaseTool):
    """
    Tool for file operations.
    
    Features:
    - Read/write text files
    - JSON/YAML/CSV processing
    - File metadata
    - Directory operations
    - Safe path handling
    - Async file operations
    """
    
    def __init__(self, 
                 allowed_directories: Optional[List[str]] = None,
                 max_file_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(
            name="FileOperations",
            description="Read, write, and process files"
        )
        self.allowed_directories = allowed_directories or [os.getcwd()]
        self.max_file_size = max_file_size
    
    def _validate_path(self, path: Union[str, Path]) -> Path:
        """
        Validate that path is within allowed directories.
        """
        path = Path(path).resolve()
        
        # Check if path is within allowed directories
        for allowed_dir in self.allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                path.relative_to(allowed_path)
                return path
            except ValueError:
                continue
        
        raise ToolError(f"Path {path} is not within allowed directories")
    
    async def read_file(self, 
                       path: Union[str, Path],
                       encoding: str = "utf-8") -> str:
        """
        Read a text file asynchronously.
        
        Args:
            path: File path
            encoding: Text encoding
            
        Returns:
            File contents
        """
        path = self._validate_path(path)
        
        if not path.exists():
            raise ToolError(f"File not found: {path}")
        
        if path.stat().st_size > self.max_file_size:
            raise ToolError(f"File too large: {path.stat().st_size} bytes")
        
        async with aiofiles.open(path, mode='r', encoding=encoding) as f:
            content = await f.read()
        
        return content
    
    async def write_file(self,
                        path: Union[str, Path],
                        content: str,
                        encoding: str = "utf-8",
                        append: bool = False) -> Dict[str, Any]:
        """
        Write to a text file asynchronously.
        
        Args:
            path: File path
            content: Content to write
            encoding: Text encoding
            append: Whether to append to existing file
            
        Returns:
            Operation result
        """
        path = self._validate_path(path)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        mode = 'a' if append else 'w'
        
        async with aiofiles.open(path, mode=mode, encoding=encoding) as f:
            await f.write(content)
        
        return {
            "path": str(path),
            "size": len(content),
            "mode": "append" if append else "write",
            "timestamp": datetime.now().isoformat()
        }
    
    async def read_json(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Read and parse a JSON file.
        """
        content = await self.read_file(path)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ToolError(f"Invalid JSON in {path}: {str(e)}")
    
    async def write_json(self,
                        path: Union[str, Path],
                        data: Union[Dict, List],
                        indent: int = 2) -> Dict[str, Any]:
        """
        Write data to a JSON file.
        """
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        return await self.write_file(path, content)
    
    async def read_yaml(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Read and parse a YAML file.
        """
        content = await self.read_file(path)
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ToolError(f"Invalid YAML in {path}: {str(e)}")
    
    async def write_yaml(self,
                        path: Union[str, Path],
                        data: Union[Dict, List]) -> Dict[str, Any]:
        """
        Write data to a YAML file.
        """
        content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
        return await self.write_file(path, content)
    
    async def read_csv(self,
                      path: Union[str, Path],
                      delimiter: str = ",",
                      has_header: bool = True) -> List[Dict[str, Any]]:
        """
        Read and parse a CSV file.
        """
        content = await self.read_file(path)
        lines = content.strip().split('\n')
        
        if not lines:
            return []
        
        reader = csv.reader(lines, delimiter=delimiter)
        
        if has_header:
            headers = next(reader)
            return [dict(zip(headers, row)) for row in reader]
        else:
            return [row for row in reader]
    
    async def write_csv(self,
                       path: Union[str, Path],
                       data: List[Dict[str, Any]],
                       delimiter: str = ",") -> Dict[str, Any]:
        """
        Write data to a CSV file.
        """
        if not data:
            return await self.write_file(path, "")
        
        # Get headers from first row
        headers = list(data[0].keys())
        
        output = []
        writer = csv.DictWriter(output, fieldnames=headers, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(data)
        
        content = '\n'.join(output)
        return await self.write_file(path, content)
    
    async def list_directory(self,
                           path: Union[str, Path] = ".",
                           pattern: Optional[str] = None,
                           recursive: bool = False) -> List[Dict[str, Any]]:
        """
        List files in a directory.
        
        Args:
            path: Directory path
            pattern: File pattern (e.g., "*.txt")
            recursive: Whether to search recursively
            
        Returns:
            List of file information
        """
        path = self._validate_path(path)
        
        if not path.is_dir():
            raise ToolError(f"Not a directory: {path}")
        
        files = []
        
        if recursive:
            file_paths = path.rglob(pattern or "*")
        else:
            file_paths = path.glob(pattern or "*")
        
        for file_path in file_paths:
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": mimetypes.guess_type(str(file_path))[0] or "unknown"
                })
        
        return sorted(files, key=lambda x: x["name"])
    
    async def get_file_info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get detailed file information.
        """
        path = self._validate_path(path)
        
        if not path.exists():
            raise ToolError(f"Path not found: {path}")
        
        stat = path.stat()
        
        info = {
            "name": path.name,
            "path": str(path),
            "size": stat.st_size,
            "is_file": path.is_file(),
            "is_directory": path.is_dir(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
        }
        
        if path.is_file():
            info["type"] = mimetypes.guess_type(str(path))[0] or "unknown"
            info["extension"] = path.suffix
        
        return info
    
    async def create_directory(self,
                             path: Union[str, Path],
                             parents: bool = True) -> Dict[str, Any]:
        """
        Create a directory.
        """
        path = self._validate_path(path)
        
        path.mkdir(parents=parents, exist_ok=True)
        
        return {
            "path": str(path),
            "created": datetime.now().isoformat()
        }
    
    async def delete_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Delete a file (with safety checks).
        """
        path = self._validate_path(path)
        
        if not path.exists():
            raise ToolError(f"File not found: {path}")
        
        if not path.is_file():
            raise ToolError(f"Not a file: {path}")
        
        # Safety check - don't delete critical files
        if path.name in ['.env', 'config.yaml', 'settings.json']:
            raise ToolError(f"Cannot delete protected file: {path.name}")
        
        path.unlink()
        
        return {
            "path": str(path),
            "deleted": datetime.now().isoformat()
        }
    
    async def search_in_files(self,
                            directory: Union[str, Path],
                            search_term: str,
                            file_pattern: str = "*",
                            case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Search for text in files.
        """
        directory = self._validate_path(directory)
        
        if not directory.is_dir():
            raise ToolError(f"Not a directory: {directory}")
        
        results = []
        
        for file_path in directory.rglob(file_pattern):
            if file_path.is_file():
                try:
                    content = await self.read_file(file_path)
                    
                    if not case_sensitive:
                        search_content = content.lower()
                        search_pattern = search_term.lower()
                    else:
                        search_content = content
                        search_pattern = search_term
                    
                    if search_pattern in search_content:
                        # Find line numbers
                        lines = content.split('\n')
                        matches = []
                        
                        for i, line in enumerate(lines, 1):
                            check_line = line.lower() if not case_sensitive else line
                            if search_pattern in check_line:
                                matches.append({
                                    "line_number": i,
                                    "line": line.strip()
                                })
                        
                        results.append({
                            "file": str(file_path),
                            "matches": matches
                        })
                        
                except Exception:
                    # Skip files that can't be read as text
                    continue
        
        return results
    
    async def process(self, input_data: str) -> str:
        """
        Process file operation request.
        """
        try:
            # Parse JSON request
            request = json.loads(input_data)
            
            operation = request.get("operation", "read")
            path = request.get("path")
            
            if not path:
                return "Error: No path specified"
            
            if operation == "read":
                content = await self.read_file(path)
                return content
            
            elif operation == "write":
                content = request.get("content", "")
                result = await self.write_file(path, content)
                return json.dumps(result)
            
            elif operation == "list":
                pattern = request.get("pattern")
                recursive = request.get("recursive", False)
                files = await self.list_directory(path, pattern, recursive)
                return json.dumps(files, indent=2)
            
            elif operation == "info":
                info = await self.get_file_info(path)
                return json.dumps(info, indent=2)
            
            elif operation == "search":
                search_term = request.get("search_term", "")
                pattern = request.get("pattern", "*")
                results = await self.search_in_files(path, search_term, pattern)
                return json.dumps(results, indent=2)
            
            else:
                return f"Error: Unknown operation '{operation}'"
                
        except json.JSONDecodeError:
            return "Error: Invalid JSON input"
        except Exception as e:
            return f"Error: {str(e)}"
