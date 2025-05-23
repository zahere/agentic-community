"""
Streaming support for real-time agent responses.

Provides streaming capabilities for LLM responses, tool outputs,
and agent reasoning steps.
"""

import asyncio
import json
from typing import AsyncIterator, Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import logging

from ..core.llm_providers import LLMRouter
from ..agents.base import BaseAgent
from ..core.base import BaseTool

logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """Types of events in the stream."""
    THOUGHT = "thought"
    TOOL_START = "tool_start"
    TOOL_OUTPUT = "tool_output"
    TOOL_END = "tool_end"
    LLM_START = "llm_start"
    LLM_TOKEN = "llm_token"
    LLM_END = "llm_end"
    ERROR = "error"
    FINAL_ANSWER = "final_answer"
    METADATA = "metadata"


@dataclass
class StreamEvent:
    """A single event in the stream."""
    type: StreamEventType
    content: Any
    timestamp: str = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
            
    def to_json(self) -> str:
        """Convert to JSON string."""
        data = {
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return json.dumps(data)
        
    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        return f"data: {self.to_json()}\n\n"


class StreamingAgent(BaseAgent):
    """Agent with streaming response capabilities."""
    
    def __init__(
        self,
        name: str = "Streaming Agent",
        llm_router: Optional[LLMRouter] = None,
        stream_thoughts: bool = True,
        stream_tool_outputs: bool = True
    ):
        super().__init__(name)
        self.llm_router = llm_router
        self.stream_thoughts = stream_thoughts
        self.stream_tool_outputs = stream_tool_outputs
        
    async def stream_process(self, task: str) -> AsyncIterator[StreamEvent]:
        """
        Process a task and stream the results.
        
        Yields StreamEvent objects for each step of the process.
        """
        # Start event
        yield StreamEvent(
            type=StreamEventType.METADATA,
            content={"agent": self.name, "task": task}
        )
        
        try:
            # Stream thought process
            if self.stream_thoughts:
                yield StreamEvent(
                    type=StreamEventType.THOUGHT,
                    content="Analyzing the task..."
                )
                
            # Determine if tools are needed
            tools_needed = await self._determine_tools_needed(task)
            
            if tools_needed:
                # Execute tools with streaming
                for tool_name in tools_needed:
                    tool = self._get_tool(tool_name)
                    if tool:
                        async for event in self._stream_tool_execution(tool, task):
                            yield event
                            
            # Stream final answer generation
            async for event in self._stream_final_answer(task):
                yield event
                
        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content=str(e),
                metadata={"error_type": type(e).__name__}
            )
            
    async def _determine_tools_needed(self, task: str) -> List[str]:
        """Determine which tools are needed for the task."""
        if not self.llm_router:
            return []
            
        tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in self.tools])
        
        prompt = f"""Given this task, which tools (if any) would be helpful?
Available tools:
{tools_desc}

Task: {task}

List only the tool names needed, one per line:"""
        
        response = await self.llm_router.complete(prompt, temperature=0.3)
        
        # Parse tool names
        tool_names = []
        for line in response.content.split('\n'):
            line = line.strip().lower()
            for tool in self.tools:
                if tool.name.lower() in line:
                    tool_names.append(tool.name)
                    break
                    
        return tool_names
        
    def _get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
        
    async def _stream_tool_execution(
        self, 
        tool: BaseTool, 
        task: str
    ) -> AsyncIterator[StreamEvent]:
        """Stream the execution of a tool."""
        # Tool start event
        yield StreamEvent(
            type=StreamEventType.TOOL_START,
            content={"tool": tool.name, "description": tool.description}
        )
        
        try:
            # Execute tool
            result = await asyncio.to_thread(tool.run, task)
            
            if self.stream_tool_outputs:
                # Stream tool output in chunks
                output_str = str(result)
                chunk_size = 100
                
                for i in range(0, len(output_str), chunk_size):
                    chunk = output_str[i:i + chunk_size]
                    yield StreamEvent(
                        type=StreamEventType.TOOL_OUTPUT,
                        content=chunk,
                        metadata={"tool": tool.name}
                    )
                    await asyncio.sleep(0.01)  # Small delay for streaming effect
            else:
                # Send complete output at once
                yield StreamEvent(
                    type=StreamEventType.TOOL_OUTPUT,
                    content=result,
                    metadata={"tool": tool.name}
                )
                
        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content=f"Tool {tool.name} failed: {str(e)}"
            )
        finally:
            # Tool end event
            yield StreamEvent(
                type=StreamEventType.TOOL_END,
                content={"tool": tool.name}
            )
            
    async def _stream_final_answer(self, task: str) -> AsyncIterator[StreamEvent]:
        """Stream the final answer generation."""
        if not self.llm_router:
            yield StreamEvent(
                type=StreamEventType.FINAL_ANSWER,
                content="Unable to generate answer without LLM."
            )
            return
            
        # LLM start event
        yield StreamEvent(
            type=StreamEventType.LLM_START,
            content={"task": "Generating final answer"}
        )
        
        prompt = f"Provide a comprehensive answer to: {task}"
        
        try:
            # Stream tokens
            token_buffer = []
            async for token in self.llm_router.stream_complete(prompt):
                yield StreamEvent(
                    type=StreamEventType.LLM_TOKEN,
                    content=token
                )
                token_buffer.append(token)
                
            # LLM end event
            complete_response = "".join(token_buffer)
            yield StreamEvent(
                type=StreamEventType.LLM_END,
                content={"response_length": len(complete_response)}
            )
            
            # Final answer event
            yield StreamEvent(
                type=StreamEventType.FINAL_ANSWER,
                content=complete_response
            )
            
        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content=f"LLM generation failed: {str(e)}"
            )


class StreamProcessor:
    """Processes and aggregates stream events."""
    
    def __init__(self):
        self.events: List[StreamEvent] = []
        self.final_answer: Optional[str] = None
        self.errors: List[str] = []
        self.tool_outputs: Dict[str, Any] = {}
        
    def process_event(self, event: StreamEvent):
        """Process a single stream event."""
        self.events.append(event)
        
        if event.type == StreamEventType.FINAL_ANSWER:
            self.final_answer = event.content
        elif event.type == StreamEventType.ERROR:
            self.errors.append(event.content)
        elif event.type == StreamEventType.TOOL_OUTPUT:
            tool_name = event.metadata.get("tool") if event.metadata else "unknown"
            if tool_name not in self.tool_outputs:
                self.tool_outputs[tool_name] = ""
            self.tool_outputs[tool_name] += str(event.content)
            
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of processed events."""
        return {
            "total_events": len(self.events),
            "final_answer": self.final_answer,
            "errors": self.errors,
            "tool_outputs": self.tool_outputs,
            "event_types": {
                event_type.value: sum(
                    1 for e in self.events if e.type == event_type
                )
                for event_type in StreamEventType
            }
        }


class BufferedStreamWriter:
    """Buffers and writes stream events efficiently."""
    
    def __init__(self, buffer_size: int = 10, flush_interval: float = 0.1):
        self.buffer: List[StreamEvent] = []
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._flush_task: Optional[asyncio.Task] = None
        
    async def write(self, event: StreamEvent) -> None:
        """Write an event to the buffer."""
        self.buffer.append(event)
        
        if len(self.buffer) >= self.buffer_size:
            await self.flush()
        elif not self._flush_task or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._delayed_flush())
            
    async def _delayed_flush(self):
        """Flush after a delay."""
        await asyncio.sleep(self.flush_interval)
        await self.flush()
        
    async def flush(self) -> List[StreamEvent]:
        """Flush the buffer and return events."""
        if not self.buffer:
            return []
            
        events = self.buffer[:]
        self.buffer.clear()
        
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            
        return events


# Utility functions for streaming
async def stream_to_string(stream: AsyncIterator[StreamEvent]) -> str:
    """Convert a stream of events to a final string result."""
    processor = StreamProcessor()
    
    async for event in stream:
        processor.process_event(event)
        
    return processor.final_answer or "No answer generated"


async def stream_with_callback(
    stream: AsyncIterator[StreamEvent],
    callback: Callable[[StreamEvent], None]
) -> str:
    """Process a stream with a callback for each event."""
    processor = StreamProcessor()
    
    async for event in stream:
        processor.process_event(event)
        callback(event)
        
    return processor.final_answer or "No answer generated"


# Example usage for WebSocket streaming
async def stream_to_websocket(
    stream: AsyncIterator[StreamEvent],
    websocket  # FastAPI WebSocket
) -> None:
    """Stream events to a WebSocket connection."""
    buffered_writer = BufferedStreamWriter()
    
    async for event in stream:
        await buffered_writer.write(event)
        
        # Check if we should flush
        events_to_send = await buffered_writer.flush()
        for buffered_event in events_to_send:
            await websocket.send_text(buffered_event.to_json())
            
    # Final flush
    remaining_events = await buffered_writer.flush()
    for event in remaining_events:
        await websocket.send_text(event.to_json())


# Example usage for Server-Sent Events (SSE)
async def stream_to_sse(stream: AsyncIterator[StreamEvent]):
    """Convert stream to Server-Sent Events format."""
    async for event in stream:
        yield event.to_sse()
