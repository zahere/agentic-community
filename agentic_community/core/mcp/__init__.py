"""
Model Context Protocol (MCP) implementation for Agentic Framework.

MCP provides a standardized way for LLMs to interact with tools and data sources,
ensuring compatibility across different AI systems.
"""

from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from abc import ABC, abstractmethod
import logging

from ..base import BaseTool
from ..exceptions import ValidationError

logger = logging.getLogger(__name__)


class MCPMessageType(Enum):
    """MCP message types."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class MCPToolParameter:
    """Defines a parameter for an MCP tool."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    pattern: Optional[str] = None


@dataclass
class MCPToolSchema:
    """Schema definition for an MCP-compatible tool."""
    name: str
    description: str
    parameters: List[MCPToolParameter]
    returns: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"


@dataclass
class MCPMessage:
    """MCP protocol message."""
    id: str
    type: MCPMessageType
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class MCPToolAdapter(ABC):
    """Abstract adapter for making tools MCP-compatible."""
    
    @abstractmethod
    def get_schema(self) -> MCPToolSchema:
        """Get the MCP schema for this tool."""
        pass
        
    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Any:
        """Execute the tool with given parameters."""
        pass


class MCPToolWrapper(MCPToolAdapter):
    """Wraps existing BaseTool instances to make them MCP-compatible."""
    
    def __init__(self, tool: BaseTool):
        self.tool = tool
        self._schema = self._generate_schema()
        
    def _generate_schema(self) -> MCPToolSchema:
        """Generate MCP schema from tool metadata."""
        # Extract parameters from tool description or docstring
        parameters = []
        
        # Simple parameter extraction - in production, use better parsing
        if hasattr(self.tool, '_run'):
            import inspect
            sig = inspect.signature(self.tool._run)
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                    
                param_type = "string"  # Default type
                if param.annotation != inspect.Parameter.empty:
                    param_type = param.annotation.__name__
                    
                parameters.append(MCPToolParameter(
                    name=param_name,
                    type=param_type,
                    description=f"Parameter {param_name}",
                    required=param.default == inspect.Parameter.empty
                ))
                
        return MCPToolSchema(
            name=self.tool.name,
            description=self.tool.description,
            parameters=parameters,
            returns={"type": "string", "description": "Tool output"}
        )
        
    def get_schema(self) -> MCPToolSchema:
        """Get the MCP schema for this tool."""
        return self._schema
        
    async def execute(self, params: Dict[str, Any]) -> Any:
        """Execute the tool with given parameters."""
        # Convert async/sync execution
        if asyncio.iscoroutinefunction(self.tool._run):
            return await self.tool._run(**params)
        else:
            return await asyncio.to_thread(self.tool._run, **params)


class MCPServer:
    """MCP server for handling tool requests."""
    
    def __init__(self):
        self.tools: Dict[str, MCPToolAdapter] = {}
        self.middleware: List[Callable] = []
        
    def register_tool(self, tool: Union[BaseTool, MCPToolAdapter], name: Optional[str] = None):
        """Register a tool with the MCP server."""
        if isinstance(tool, BaseTool):
            adapter = MCPToolWrapper(tool)
            tool_name = name or tool.name
        else:
            adapter = tool
            tool_name = name or adapter.get_schema().name
            
        self.tools[tool_name] = adapter
        logger.info(f"Registered MCP tool: {tool_name}")
        
    def add_middleware(self, middleware: Callable):
        """Add middleware for request processing."""
        self.middleware.append(middleware)
        
    async def handle_request(self, message: MCPMessage) -> MCPMessage:
        """Handle an MCP request."""
        try:
            # Apply middleware
            for mw in self.middleware:
                message = await mw(message)
                
            if message.method == "tools/list":
                # List available tools
                result = {
                    "tools": [
                        {
                            "name": name,
                            "schema": adapter.get_schema().__dict__
                        }
                        for name, adapter in self.tools.items()
                    ]
                }
                
            elif message.method == "tools/execute":
                # Execute a tool
                tool_name = message.params.get("tool")
                if tool_name not in self.tools:
                    raise ValueError(f"Unknown tool: {tool_name}")
                    
                tool_params = message.params.get("parameters", {})
                result = await self.tools[tool_name].execute(tool_params)
                
            else:
                raise ValueError(f"Unknown method: {message.method}")
                
            return MCPMessage(
                id=message.id,
                type=MCPMessageType.RESPONSE,
                result=result
            )
            
        except Exception as e:
            logger.error(f"MCP request error: {e}")
            return MCPMessage(
                id=message.id,
                type=MCPMessageType.ERROR,
                error={
                    "code": -32603,
                    "message": str(e)
                }
            )


class MCPClient:
    """Client for interacting with MCP servers."""
    
    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url
        self.local_server = MCPServer() if not server_url else None
        
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server."""
        message = MCPMessage(
            id="1",
            type=MCPMessageType.REQUEST,
            method="tools/list"
        )
        
        if self.local_server:
            response = await self.local_server.handle_request(message)
        else:
            # TODO: Implement remote server communication
            raise NotImplementedError("Remote MCP servers not yet supported")
            
        return response.result.get("tools", [])
        
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a tool on the server."""
        message = MCPMessage(
            id="2",
            type=MCPMessageType.REQUEST,
            method="tools/execute",
            params={
                "tool": tool_name,
                "parameters": parameters
            }
        )
        
        if self.local_server:
            response = await self.local_server.handle_request(message)
        else:
            # TODO: Implement remote server communication
            raise NotImplementedError("Remote MCP servers not yet supported")
            
        if response.type == MCPMessageType.ERROR:
            raise Exception(response.error.get("message"))
            
        return response.result


# Convenience functions
def create_mcp_server() -> MCPServer:
    """Create a new MCP server instance."""
    return MCPServer()


def wrap_tool_for_mcp(tool: BaseTool) -> MCPToolAdapter:
    """Wrap a standard tool to be MCP-compatible."""
    return MCPToolWrapper(tool)


# Example custom MCP tool
class MCPSearchTool(MCPToolAdapter):
    """Example of a native MCP tool implementation."""
    
    def get_schema(self) -> MCPToolSchema:
        return MCPToolSchema(
            name="mcp_search",
            description="Search the web using MCP protocol",
            parameters=[
                MCPToolParameter(
                    name="query",
                    type="string",
                    description="Search query"
                ),
                MCPToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=10
                )
            ],
            returns={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "snippet": {"type": "string"}
                    }
                }
            },
            examples=[
                {
                    "params": {"query": "AI agents", "max_results": 5},
                    "result": [{"title": "...", "url": "...", "snippet": "..."}]
                }
            ]
        )
        
    async def execute(self, params: Dict[str, Any]) -> Any:
        # Implementation would go here
        query = params.get("query")
        max_results = params.get("max_results", 10)
        
        # Placeholder implementation
        return [
            {
                "title": f"Result for {query}",
                "url": "https://example.com",
                "snippet": "Example search result"
            }
        ]
