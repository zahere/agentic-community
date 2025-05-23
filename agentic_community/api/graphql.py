"""
GraphQL API for Agentic Framework

This module provides a GraphQL interface for the framework, offering
a more flexible and efficient alternative to REST.
"""

from typing import List, Optional, Dict, Any, AsyncGenerator
import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info
from strawberry.dataloader import DataLoader
from datetime import datetime
import asyncio
import uuid

from agentic_community.core.base.agent import BaseAgent
from agentic_community.core.base.tool import BaseTool
from agentic_community.agents import SimpleAgent, MockAgent
from agentic_community.core.telemetry import trace


# GraphQL Types

@strawberry.type
class AgentType:
    """GraphQL type for agents"""
    id: str
    name: str
    type: str
    description: str
    status: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None


@strawberry.type
class ToolType:
    """GraphQL type for tools"""
    id: str
    name: str
    description: str
    category: str
    enabled: bool
    parameters: Optional[Dict[str, Any]] = None


@strawberry.type
class ExecutionResult:
    """GraphQL type for execution results"""
    id: str
    agent_id: str
    input: str
    output: str
    success: bool
    duration_ms: float
    created_at: datetime
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@strawberry.type
class WorkflowType:
    """GraphQL type for workflows"""
    id: str
    name: str
    description: str
    status: str
    steps: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


@strawberry.type
class MetricType:
    """GraphQL type for metrics"""
    name: str
    value: float
    unit: str
    labels: Dict[str, str]
    timestamp: datetime


@strawberry.type
class TraceType:
    """GraphQL type for traces"""
    trace_id: str
    span_id: str
    operation_name: str
    duration_ms: float
    status: str
    attributes: Dict[str, Any]


# Input Types

@strawberry.input
class CreateAgentInput:
    """Input for creating an agent"""
    name: str
    type: str
    description: Optional[str] = None
    tools: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None


@strawberry.input
class ExecuteAgentInput:
    """Input for executing an agent"""
    agent_id: str
    input: str
    context: Optional[Dict[str, Any]] = None
    stream: bool = False


@strawberry.input
class QueryFilter:
    """Generic query filter"""
    field: str
    operator: str = "eq"  # eq, ne, gt, lt, gte, lte, contains
    value: Any


@strawberry.input
class PaginationInput:
    """Pagination parameters"""
    offset: int = 0
    limit: int = 20
    sort_by: Optional[str] = None
    sort_order: str = "asc"


# Subscriptions

@strawberry.type
class AgentUpdate:
    """Real-time agent updates"""
    agent_id: str
    event_type: str  # created, updated, deleted, executed
    data: Dict[str, Any]
    timestamp: datetime


# Query Root

@strawberry.type
class Query:
    """GraphQL queries"""
    
    @strawberry.field
    @trace("graphql.query.agents")
    async def agents(
        self,
        filters: Optional[List[QueryFilter]] = None,
        pagination: Optional[PaginationInput] = None
    ) -> List[AgentType]:
        """List all agents with optional filtering and pagination"""
        # Simulate agent retrieval
        agents = [
            AgentType(
                id="agent1",
                name="SimpleAgent",
                type="simple",
                description="A simple agent",
                status="active",
                created_at=datetime.now()
            ),
            AgentType(
                id="agent2",
                name="AdvancedAgent",
                type="advanced",
                description="An advanced agent",
                status="active",
                created_at=datetime.now()
            )
        ]
        
        # Apply filters and pagination (simplified)
        if pagination:
            start = pagination.offset
            end = start + pagination.limit
            agents = agents[start:end]
        
        return agents
    
    @strawberry.field
    @trace("graphql.query.agent")
    async def agent(self, id: str) -> Optional[AgentType]:
        """Get a specific agent by ID"""
        # Simulate agent retrieval
        if id == "agent1":
            return AgentType(
                id="agent1",
                name="SimpleAgent",
                type="simple",
                description="A simple agent",
                status="active",
                created_at=datetime.now()
            )
        return None
    
    @strawberry.field
    @trace("graphql.query.tools")
    async def tools(
        self,
        category: Optional[str] = None,
        enabled: Optional[bool] = None
    ) -> List[ToolType]:
        """List available tools"""
        tools = [
            ToolType(
                id="tool1",
                name="SearchTool",
                description="Search the web",
                category="research",
                enabled=True
            ),
            ToolType(
                id="tool2",
                name="CalculatorTool",
                description="Perform calculations",
                category="utility",
                enabled=True
            )
        ]
        
        # Filter by category
        if category:
            tools = [t for t in tools if t.category == category]
        
        # Filter by enabled status
        if enabled is not None:
            tools = [t for t in tools if t.enabled == enabled]
        
        return tools
    
    @strawberry.field
    @trace("graphql.query.executions")
    async def executions(
        self,
        agent_id: Optional[str] = None,
        success: Optional[bool] = None,
        pagination: Optional[PaginationInput] = None
    ) -> List[ExecutionResult]:
        """Query execution history"""
        # Simulate execution history
        executions = [
            ExecutionResult(
                id="exec1",
                agent_id="agent1",
                input="Hello",
                output="Hello! How can I help you?",
                success=True,
                duration_ms=150.5,
                created_at=datetime.now()
            )
        ]
        
        # Apply filters
        if agent_id:
            executions = [e for e in executions if e.agent_id == agent_id]
        
        if success is not None:
            executions = [e for e in executions if e.success == success]
        
        return executions
    
    @strawberry.field
    @trace("graphql.query.workflows")
    async def workflows(
        self,
        status: Optional[str] = None
    ) -> List[WorkflowType]:
        """List workflows"""
        workflows = [
            WorkflowType(
                id="wf1",
                name="Data Processing Workflow",
                description="Process data through multiple steps",
                status="completed",
                steps=[
                    {"name": "Load Data", "status": "completed"},
                    {"name": "Transform", "status": "completed"},
                    {"name": "Save Results", "status": "completed"}
                ],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        
        if status:
            workflows = [w for w in workflows if w.status == status]
        
        return workflows
    
    @strawberry.field
    @trace("graphql.query.metrics")
    async def metrics(
        self,
        names: Optional[List[str]] = None,
        time_range: int = 3600
    ) -> List[MetricType]:
        """Query metrics"""
        metrics = [
            MetricType(
                name="agent_invocations_total",
                value=1234,
                unit="1",
                labels={"agent": "SimpleAgent"},
                timestamp=datetime.now()
            ),
            MetricType(
                name="agent_duration_seconds",
                value=0.156,
                unit="s",
                labels={"agent": "SimpleAgent"},
                timestamp=datetime.now()
            )
        ]
        
        if names:
            metrics = [m for m in metrics if m.name in names]
        
        return metrics
    
    @strawberry.field
    @trace("graphql.query.traces")
    async def traces(
        self,
        trace_id: Optional[str] = None,
        operation: Optional[str] = None,
        limit: int = 100
    ) -> List[TraceType]:
        """Query distributed traces"""
        traces = [
            TraceType(
                trace_id="trace123",
                span_id="span456",
                operation_name="agent.process",
                duration_ms=245.3,
                status="ok",
                attributes={"agent": "SimpleAgent", "input_length": 15}
            )
        ]
        
        if operation:
            traces = [t for t in traces if t.operation_name == operation]
        
        return traces[:limit]


# Mutation Root

@strawberry.type
class Mutation:
    """GraphQL mutations"""
    
    @strawberry.mutation
    @trace("graphql.mutation.create_agent")
    async def create_agent(self, input: CreateAgentInput) -> AgentType:
        """Create a new agent"""
        # Create agent instance based on type
        agent_id = f"agent_{datetime.now().timestamp()}"
        
        return AgentType(
            id=agent_id,
            name=input.name,
            type=input.type,
            description=input.description or "",
            status="active",
            created_at=datetime.now(),
            metadata=input.config
        )
    
    @strawberry.mutation
    @trace("graphql.mutation.execute_agent")
    async def execute_agent(self, input: ExecuteAgentInput) -> ExecutionResult:
        """Execute an agent"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Simulate agent execution
            output = f"Processed: {input.input}"
            success = True
            error = None
        except Exception as e:
            output = ""
            success = False
            error = str(e)
        
        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return ExecutionResult(
            id=f"exec_{datetime.now().timestamp()}",
            agent_id=input.agent_id,
            input=input.input,
            output=output,
            success=success,
            duration_ms=duration_ms,
            created_at=datetime.now(),
            error=error,
            metadata=input.context
        )
    
    @strawberry.mutation
    @trace("graphql.mutation.update_agent")
    async def update_agent(
        self,
        id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentType:
        """Update an agent"""
        # Simulate agent update
        return AgentType(
            id=id,
            name=name or "UpdatedAgent",
            type="simple",
            description=description or "Updated description",
            status="active",
            created_at=datetime.now(),
            metadata=metadata
        )
    
    @strawberry.mutation
    @trace("graphql.mutation.delete_agent")
    async def delete_agent(self, id: str) -> bool:
        """Delete an agent"""
        # Simulate agent deletion
        return True
    
    @strawberry.mutation
    @trace("graphql.mutation.create_workflow")
    async def create_workflow(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]]
    ) -> WorkflowType:
        """Create a workflow"""
        return WorkflowType(
            id=f"wf_{datetime.now().timestamp()}",
            name=name,
            description=description,
            status="pending",
            steps=steps,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    @strawberry.mutation
    @trace("graphql.mutation.start_workflow")
    async def start_workflow(self, id: str) -> WorkflowType:
        """Start a workflow"""
        return WorkflowType(
            id=id,
            name="Started Workflow",
            description="Workflow has been started",
            status="running",
            steps=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )


# Subscription Root

@strawberry.type
class Subscription:
    """GraphQL subscriptions for real-time updates"""
    
    @strawberry.subscription
    async def agent_updates(
        self,
        agent_id: Optional[str] = None
    ) -> AsyncGenerator[AgentUpdate, None]:
        """Subscribe to agent updates"""
        # Simulate real-time updates
        while True:
            await asyncio.sleep(5)  # Send update every 5 seconds
            
            update = AgentUpdate(
                agent_id=agent_id or "agent1",
                event_type="executed",
                data={
                    "input": "Sample input",
                    "output": "Sample output",
                    "duration_ms": 123.4
                },
                timestamp=datetime.now()
            )
            
            yield update
    
    @strawberry.subscription
    async def metrics_stream(
        self,
        metric_names: Optional[List[str]] = None,
        interval: int = 1
    ) -> AsyncGenerator[List[MetricType], None]:
        """Stream metrics in real-time"""
        while True:
            await asyncio.sleep(interval)
            
            # Generate sample metrics
            metrics = [
                MetricType(
                    name="agent_invocations_total",
                    value=asyncio.get_event_loop().time() % 100,
                    unit="1",
                    labels={"agent": "SimpleAgent"},
                    timestamp=datetime.now()
                )
            ]
            
            if metric_names:
                metrics = [m for m in metrics if m.name in metric_names]
            
            yield metrics
    
    @strawberry.subscription
    async def workflow_status(
        self,
        workflow_id: str
    ) -> AsyncGenerator[WorkflowType, None]:
        """Subscribe to workflow status updates"""
        statuses = ["running", "completed"]
        status_index = 0
        
        while status_index < len(statuses):
            await asyncio.sleep(3)
            
            workflow = WorkflowType(
                id=workflow_id,
                name="Sample Workflow",
                description="Workflow with status updates",
                status=statuses[status_index],
                steps=[],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            yield workflow
            status_index += 1


# DataLoader for efficient data fetching

async def load_agents(agent_ids: List[str]) -> List[Optional[AgentType]]:
    """Batch load agents by IDs"""
    # Simulate batch loading
    agents = []
    for agent_id in agent_ids:
        if agent_id == "agent1":
            agents.append(AgentType(
                id="agent1",
                name="SimpleAgent",
                type="simple",
                description="A simple agent",
                status="active",
                created_at=datetime.now()
            ))
        else:
            agents.append(None)
    return agents


# Create GraphQL Schema

schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)


# Create GraphQL Router for FastAPI

def create_graphql_router(
    path: str = "/graphql",
    include_graphiql: bool = True
) -> GraphQLRouter:
    """Create a GraphQL router for FastAPI integration"""
    
    # Create DataLoader
    agent_loader = DataLoader(load_fn=load_agents)
    
    # Create context function
    async def get_context() -> Dict[str, Any]:
        return {
            "agent_loader": agent_loader,
            "request_id": str(uuid.uuid4())
        }
    
    # Create router
    graphql_router = GraphQLRouter(
        schema,
        path=path,
        graphiql=include_graphiql,
        context_getter=get_context
    )
    
    return graphql_router


# Federation support for microservices

@strawberry.federation.type(keys=["id"])
class FederatedAgent:
    """Federated agent type for GraphQL Federation"""
    id: strawberry.ID
    
    @strawberry.field
    def name(self) -> str:
        return f"Agent {self.id}"
    
    @classmethod
    def resolve_reference(cls, id: strawberry.ID) -> "FederatedAgent":
        return cls(id=id)


# Custom scalars

@strawberry.scalar(
    serialize=lambda value: value.isoformat(),
    parse_value=lambda value: datetime.fromisoformat(value)
)
class DateTime:
    """Custom DateTime scalar"""
    pass


# Example usage

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    # Create FastAPI app
    app = FastAPI(title="Agentic Framework GraphQL API")
    
    # Add GraphQL router
    graphql_router = create_graphql_router()
    app.include_router(graphql_router)
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8001)
