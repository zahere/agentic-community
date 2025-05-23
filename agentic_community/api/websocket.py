"""
WebSocket support for real-time agent communication.

Provides WebSocket endpoints for streaming agent responses and handling
long-running tasks with real-time updates.
"""

import json
import asyncio
from typing import Dict, Any, Optional, Set
from datetime import datetime
import logging

from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

from ..agents import SimpleAgent
from ..core.exceptions import AgentError, ToolError
from .auth import verify_token

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for agents."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.agent_sessions: Dict[str, SimpleAgent] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
        
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.agent_sessions:
            del self.agent_sessions[client_id]
        logger.info(f"Client {client_id} disconnected")
        
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_json(message)
            
    async def broadcast(self, message: Dict[str, Any], exclude: Optional[Set[str]] = None):
        """Broadcast a message to all connected clients."""
        exclude = exclude or set()
        disconnected = []
        
        for client_id, websocket in self.active_connections.items():
            if client_id not in exclude:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to {client_id}: {e}")
                    disconnected.append(client_id)
                    
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)
            
    def get_or_create_agent(self, client_id: str, agent_name: str = "Assistant") -> SimpleAgent:
        """Get or create an agent for a client session."""
        if client_id not in self.agent_sessions:
            self.agent_sessions[client_id] = SimpleAgent(agent_name)
        return self.agent_sessions[client_id]


# Global connection manager
manager = ConnectionManager()

# Security for WebSocket
security = HTTPBearer()


async def get_current_user_ws(websocket: WebSocket):
    """Extract and verify JWT token from WebSocket connection."""
    try:
        # Get token from query params or headers
        token = websocket.query_params.get("token")
        if not token:
            # Try to get from headers
            auth_header = websocket.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                
        if not token:
            await websocket.close(code=1008, reason="Missing authentication token")
            return None
            
        # Verify token
        payload = verify_token(token)
        return payload.get("sub")
    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        await websocket.close(code=1008, reason="Invalid authentication token")
        return None


async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time agent communication.
    
    Message format:
    {
        "type": "execute" | "add_tool" | "get_state" | "clear_memory",
        "data": {
            "task": "...",  # for execute
            "tool_name": "...",  # for add_tool
            ...
        }
    }
    
    Response format:
    {
        "type": "result" | "error" | "progress" | "state",
        "data": {...},
        "timestamp": "..."
    }
    """
    # Authenticate
    user_id = await get_current_user_ws(websocket)
    if not user_id:
        return
        
    # Connect
    await manager.connect(websocket, client_id)
    agent = manager.get_or_create_agent(client_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message_type = data.get("type")
            message_data = data.get("data", {})
            
            response = {
                "timestamp": datetime.utcnow().isoformat()
            }
            
            try:
                if message_type == "execute":
                    # Execute task with progress updates
                    task = message_data.get("task")
                    if not task:
                        raise ValueError("Task is required")
                        
                    # Send progress update
                    await manager.send_message(client_id, {
                        "type": "progress",
                        "data": {"status": "starting", "task": task},
                        "timestamp": response["timestamp"]
                    })
                    
                    # Execute task
                    result = await asyncio.to_thread(agent.run, task)
                    
                    response["type"] = "result"
                    response["data"] = {
                        "result": result,
                        "task": task
                    }
                    
                elif message_type == "add_tool":
                    # Add tool to agent
                    tool_name = message_data.get("tool_name")
                    if not tool_name:
                        raise ValueError("Tool name is required")
                        
                    # Import and add tool dynamically
                    # This is a simplified version - in production, validate tool names
                    from .. import tools
                    tool_class = getattr(tools, tool_name, None)
                    if not tool_class:
                        raise ValueError(f"Unknown tool: {tool_name}")
                        
                    tool = tool_class()
                    agent.add_tool(tool)
                    
                    response["type"] = "result"
                    response["data"] = {
                        "message": f"Tool {tool_name} added successfully",
                        "tools": [t.name for t in agent.tools]
                    }
                    
                elif message_type == "get_state":
                    # Get agent state
                    response["type"] = "state"
                    response["data"] = {
                        "name": agent.name,
                        "tools": [t.name for t in agent.tools],
                        "memory_size": len(agent.memory.short_term_memory) if hasattr(agent, 'memory') else 0,
                        "conversation_count": agent.conversation_count if hasattr(agent, 'conversation_count') else 0
                    }
                    
                elif message_type == "clear_memory":
                    # Clear agent memory
                    if hasattr(agent, 'memory'):
                        agent.memory.clear()
                    
                    response["type"] = "result"
                    response["data"] = {"message": "Memory cleared successfully"}
                    
                else:
                    raise ValueError(f"Unknown message type: {message_type}")
                    
            except (AgentError, ToolError) as e:
                response["type"] = "error"
                response["data"] = {
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                response["type"] = "error"
                response["data"] = {
                    "error": str(e),
                    "error_type": "UnexpectedError"
                }
                
            # Send response
            await manager.send_message(client_id, response)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        manager.disconnect(client_id)


async def broadcast_system_message(message: str, message_type: str = "info"):
    """Broadcast a system message to all connected clients."""
    await manager.broadcast({
        "type": "system",
        "data": {
            "message": message,
            "message_type": message_type
        },
        "timestamp": datetime.utcnow().isoformat()
    })


def get_active_connections() -> Dict[str, Any]:
    """Get information about active connections."""
    return {
        "total_connections": len(manager.active_connections),
        "active_clients": list(manager.active_connections.keys()),
        "active_agents": {
            client_id: {
                "name": agent.name,
                "tools": [t.name for t in agent.tools]
            }
            for client_id, agent in manager.agent_sessions.items()
        }
    }
