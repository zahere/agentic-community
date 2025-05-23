"""
FastAPI Service - Community Edition
REST API for Agentic AI Framework
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from community import SimpleAgent, SearchTool, CalculatorTool, TextTool
from agentic_community.core.licensing import get_license_manager
from agentic_community.core.utils import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Security
security = HTTPBearer()


# Request/Response Models
class AgentRequest(BaseModel):
    """Request to create an agent."""
    name: str = Field(description="Agent name")
    openai_api_key: str = Field(description="OpenAI API key")


class TaskRequest(BaseModel):
    """Request to execute a task."""
    task: str = Field(description="Task to execute")
    agent_id: str = Field(description="Agent ID")
    tools: Optional[List[str]] = Field(default=None, description="Tools to use")


class TaskResponse(BaseModel):
    """Response from task execution."""
    agent_id: str
    task: str
    result: str
    timestamp: datetime
    confidence: Optional[float] = None
    reasoning_type: Optional[str] = None


class AgentInfo(BaseModel):
    """Agent information."""
    id: str
    name: str
    created_at: datetime
    tasks_completed: int
    edition: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    edition: str
    features: Dict[str, bool]


# Global state (in production, use proper database)
agents: Dict[str, SimpleAgent] = {}
agent_stats: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting Agentic AI Framework API")
    # Startup tasks
    yield
    # Shutdown tasks
    logger.info("Shutting down Agentic AI Framework API")


# Create FastAPI app
app = FastAPI(
    title="Agentic AI Framework API",
    description="Build autonomous AI agents - Community Edition",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper functions
def get_available_tools() -> Dict[str, Any]:
    """Get available tools based on edition."""
    return {
        "search": SearchTool,
        "calculator": CalculatorTool,
        "text": TextTool
    }


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify API token (simplified for demo)."""
    # In production, implement proper authentication
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return token


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    license_manager = get_license_manager()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        edition=license_manager.get_edition(),
        features={
            "basic_reasoning": license_manager.check_feature("basic_reasoning"),
            "advanced_reasoning": license_manager.check_feature("advanced_reasoning"),
            "multi_agent": license_manager.check_feature("multi_agent")
        }
    )


@app.post("/agents", response_model=AgentInfo)
async def create_agent(
    request: AgentRequest,
    token: str = Depends(verify_token)
):
    """Create a new agent."""
    try:
        # Check agent limit for community edition
        license_manager = get_license_manager()
        limits = license_manager.get_limits()
        
        if len(agents) >= limits.get("max_agents", 1):
            raise HTTPException(
                status_code=403, 
                detail=f"Agent limit reached ({limits['max_agents']}). Upgrade to Enterprise for unlimited agents."
            )
        
        # Create agent
        agent = SimpleAgent(request.name, openai_api_key=request.openai_api_key)
        
        # Generate ID
        agent_id = f"agent_{len(agents) + 1}_{datetime.now().timestamp()}"
        
        # Store agent
        agents[agent_id] = agent
        agent_stats[agent_id] = {
            "created_at": datetime.now(),
            "tasks_completed": 0
        }
        
        logger.info(f"Created agent: {agent_id}")
        
        return AgentInfo(
            id=agent_id,
            name=request.name,
            created_at=agent_stats[agent_id]["created_at"],
            tasks_completed=0,
            edition=license_manager.get_edition()
        )
        
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/agents", response_model=List[AgentInfo])
async def list_agents(token: str = Depends(verify_token)):
    """List all agents."""
    license_manager = get_license_manager()
    
    agent_list = []
    for agent_id, agent in agents.items():
        stats = agent_stats.get(agent_id, {})
        agent_list.append(AgentInfo(
            id=agent_id,
            name=agent.config.name,
            created_at=stats.get("created_at", datetime.now()),
            tasks_completed=stats.get("tasks_completed", 0),
            edition=license_manager.get_edition()
        ))
        
    return agent_list


@app.get("/agents/{agent_id}", response_model=AgentInfo)
async def get_agent(
    agent_id: str,
    token: str = Depends(verify_token)
):
    """Get agent details."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    agent = agents[agent_id]
    stats = agent_stats.get(agent_id, {})
    license_manager = get_license_manager()
    
    return AgentInfo(
        id=agent_id,
        name=agent.config.name,
        created_at=stats.get("created_at", datetime.now()),
        tasks_completed=stats.get("tasks_completed", 0),
        edition=license_manager.get_edition()
    )


@app.post("/tasks", response_model=TaskResponse)
async def execute_task(
    request: TaskRequest,
    token: str = Depends(verify_token)
):
    """Execute a task with an agent."""
    # Validate agent exists
    if request.agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    agent = agents[request.agent_id]
    
    try:
        # Add tools if requested
        if request.tools:
            available_tools = get_available_tools()
            license_manager = get_license_manager()
            limits = license_manager.get_limits()
            
            # Check tool limit
            if len(request.tools) > limits.get("max_tools", 3):
                raise HTTPException(
                    status_code=403,
                    detail=f"Tool limit exceeded ({limits['max_tools']}). Upgrade to Enterprise for unlimited tools."
                )
                
            # Add tools to agent
            for tool_name in request.tools:
                if tool_name in available_tools:
                    tool_class = available_tools[tool_name]
                    agent.add_tool(tool_class())
                    
        # Execute task
        logger.info(f"Executing task for agent {request.agent_id}: {request.task}")
        result = agent.run(request.task)
        
        # Get reasoning info
        thoughts = agent.think(request.task)
        
        # Update stats
        agent_stats[request.agent_id]["tasks_completed"] += 1
        
        return TaskResponse(
            agent_id=request.agent_id,
            task=request.task,
            result=result,
            timestamp=datetime.now(),
            confidence=thoughts.get("confidence", 0.8),
            reasoning_type=thoughts.get("reasoning_type", "sequential")
        )
        
    except Exception as e:
        logger.error(f"Error executing task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    token: str = Depends(verify_token)
):
    """Delete an agent."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    del agents[agent_id]
    if agent_id in agent_stats:
        del agent_stats[agent_id]
        
    logger.info(f"Deleted agent: {agent_id}")
    
    return {"message": f"Agent {agent_id} deleted successfully"}


@app.get("/tools", response_model=Dict[str, Any])
async def list_tools(token: str = Depends(verify_token)):
    """List available tools."""
    tools = get_available_tools()
    license_manager = get_license_manager()
    limits = license_manager.get_limits()
    
    tool_info = {}
    for name, tool_class in tools.items():
        tool = tool_class()
        tool_info[name] = {
            "name": tool.config.name,
            "description": tool.config.description,
            "available": True
        }
        
    return {
        "tools": tool_info,
        "max_tools_per_agent": limits.get("max_tools", 3),
        "edition": license_manager.get_edition()
    }


@app.get("/license", response_model=Dict[str, Any])
async def get_license_info(token: str = Depends(verify_token)):
    """Get license information."""
    license_manager = get_license_manager()
    
    return {
        "edition": license_manager.get_edition(),
        "limits": license_manager.get_limits(),
        "features": {
            "basic_reasoning": license_manager.check_feature("basic_reasoning"),
            "advanced_reasoning": license_manager.check_feature("advanced_reasoning"),
            "self_reflection": license_manager.check_feature("self_reflection"),
            "multi_agent": license_manager.check_feature("multi_agent"),
            "all_llm_providers": license_manager.check_feature("all_llm_providers")
        },
        "upgrade_url": "https://agentic-ai.com/enterprise"
    }


# CLI entry point
def main():
    """Run the API server."""
    import os
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "community.api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
