"""
ReAct (Reasoning + Acting) Agent Implementation.

ReAct agents use a thought-action-observation loop to solve complex problems
through step-by-step reasoning and tool usage.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from .base import BaseAgent
from ..core.base import BaseTool
from ..core.exceptions import AgentError, ReasoningError
from ..core.llm_providers import LLMRouter, LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class ReActStepType(Enum):
    """Types of steps in ReAct reasoning."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    ANSWER = "answer"


@dataclass
class ReActStep:
    """A single step in the ReAct reasoning process."""
    step_type: ReActStepType
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.step_type.value,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "timestamp": self.timestamp
        }


@dataclass
class ReActTrace:
    """Complete trace of a ReAct reasoning process."""
    question: str
    steps: List[ReActStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    
    def add_thought(self, thought: str):
        self.steps.append(ReActStep(ReActStepType.THOUGHT, thought))
        
    def add_action(self, action: str, tool_name: str, tool_input: str):
        self.steps.append(ReActStep(
            ReActStepType.ACTION, 
            action, 
            tool_name=tool_name,
            tool_input=tool_input
        ))
        
    def add_observation(self, observation: str):
        self.steps.append(ReActStep(ReActStepType.OBSERVATION, observation))
        
    def set_answer(self, answer: str):
        self.final_answer = answer
        self.steps.append(ReActStep(ReActStepType.ANSWER, answer))
        self.success = True


class ReActAgent(BaseAgent):
    """
    ReAct Agent that implements the Reasoning + Acting paradigm.
    
    The agent follows this pattern:
    1. Thought: Reason about the current state
    2. Action: Decide on and execute an action
    3. Observation: Observe the result
    4. Repeat until task is complete
    """
    
    def __init__(
        self, 
        name: str = "ReAct Agent",
        llm_router: Optional[LLMRouter] = None,
        max_steps: int = 10,
        enable_reflection: bool = True
    ):
        super().__init__(name)
        self.llm_router = llm_router
        self.max_steps = max_steps
        self.enable_reflection = enable_reflection
        self.traces: List[ReActTrace] = []
        
    def _create_react_prompt(self, question: str, trace: ReActTrace) -> str:
        """Create the ReAct prompt with current trace."""
        tools_description = self._get_tools_description()
        
        prompt = f"""You are a ReAct agent that solves problems through reasoning and acting.

Available tools:
{tools_description}

Use the following format EXACTLY:

Thought: [your reasoning about what to do next]
Action: [the action to take, should be one of [{', '.join([tool.name for tool in self.tools])}]]
Action Input: [the input to the action]
Observation: [the result of the action]
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: [the final answer to the original question]

Question: {question}

"""
        # Add previous steps to prompt
        for step in trace.steps:
            if step.step_type == ReActStepType.THOUGHT:
                prompt += f"Thought: {step.content}\n"
            elif step.step_type == ReActStepType.ACTION:
                prompt += f"Action: {step.tool_name}\n"
                prompt += f"Action Input: {step.tool_input}\n"
            elif step.step_type == ReActStepType.OBSERVATION:
                prompt += f"Observation: {step.content}\n"
                
        return prompt
        
    def _get_tools_description(self) -> str:
        """Get formatted description of available tools."""
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)
        
    def _parse_llm_output(self, output: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Parse LLM output to extract thought, action, and action input."""
        # Extract thought
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)", output, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        # Check for final answer
        final_answer_match = re.search(r"Final Answer:\s*(.*?)$", output, re.DOTALL)
        if final_answer_match:
            return thought, None, final_answer_match.group(1).strip()
            
        # Extract action and input
        action_match = re.search(r"Action:\s*(.*?)(?=Action Input:|$)", output, re.DOTALL)
        action = action_match.group(1).strip() if action_match else None
        
        input_match = re.search(r"Action Input:\s*(.*?)(?=Observation:|$)", output, re.DOTALL)
        action_input = input_match.group(1).strip() if input_match else None
        
        return thought, action, action_input
        
    def _execute_action(self, tool_name: str, tool_input: str) -> str:
        """Execute an action using the specified tool."""
        # Find the tool
        tool = None
        for t in self.tools:
            if t.name.lower() == tool_name.lower():
                tool = t
                break
                
        if not tool:
            return f"Error: Tool '{tool_name}' not found. Available tools: {[t.name for t in self.tools]}"
            
        try:
            # Execute the tool
            result = tool.run(tool_input)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
            
    def _reflect_on_trace(self, trace: ReActTrace) -> str:
        """Reflect on the reasoning trace to improve future attempts."""
        if not self.enable_reflection or not trace.steps:
            return ""
            
        reflection_prompt = f"""Analyze this reasoning trace and provide insights:

Question: {trace.question}

Steps taken:
"""
        for i, step in enumerate(trace.steps):
            reflection_prompt += f"{i+1}. {step.step_type.value}: {step.content}\n"
            
        reflection_prompt += """

Provide a brief analysis of:
1. What went well in the reasoning process
2. What could be improved
3. Any patterns or insights noticed
"""
        
        # Use LLM to generate reflection
        if self.llm_router:
            try:
                response = self.llm_router.chat([
                    {"role": "system", "content": "You are an AI assistant analyzing reasoning traces."},
                    {"role": "user", "content": reflection_prompt}
                ])
                return response.content
            except Exception as e:
                logger.error(f"Reflection failed: {e}")
                
        return ""
        
    def process(self, task: str) -> str:
        """
        Process a task using ReAct reasoning.
        
        Args:
            task: The task or question to solve
            
        Returns:
            The final answer or result
        """
        trace = ReActTrace(question=task)
        
        try:
            for step_num in range(self.max_steps):
                # Create prompt with current trace
                prompt = self._create_react_prompt(task, trace)
                
                # Get LLM response
                if self.llm_router:
                    response = self.llm_router.chat([
                        {"role": "system", "content": "You are a ReAct agent. Follow the format exactly."},
                        {"role": "user", "content": prompt}
                    ])
                    output = response.content
                else:
                    # Fallback for testing without LLM
                    output = "Thought: I need to search for information.\nAction: search\nAction Input: " + task
                    
                # Parse the output
                thought, action, action_input = self._parse_llm_output(output)
                
                # Add thought to trace
                if thought:
                    trace.add_thought(thought)
                    
                # Check if we have a final answer
                if action is None and action_input:  # action_input contains final answer
                    trace.set_answer(action_input)
                    break
                    
                # Execute action if present
                if action and action_input:
                    trace.add_action(output, action, action_input)
                    observation = self._execute_action(action, action_input)
                    trace.add_observation(observation)
                    
                # Safety check to prevent infinite loops
                if step_num == self.max_steps - 1:
                    trace.error = "Maximum steps reached without finding answer"
                    break
                    
            # Perform reflection if enabled
            if self.enable_reflection and trace.steps:
                reflection = self._reflect_on_trace(trace)
                if reflection:
                    logger.info(f"Reflection: {reflection}")
                    
            # Store trace for analysis
            self.traces.append(trace)
            
            if trace.success and trace.final_answer:
                return trace.final_answer
            else:
                return f"Failed to find answer. Error: {trace.error or 'Unknown error'}"
                
        except Exception as e:
            trace.error = str(e)
            self.traces.append(trace)
            raise ReasoningError(f"ReAct reasoning failed: {e}")
            
    def get_last_trace(self) -> Optional[ReActTrace]:
        """Get the most recent reasoning trace."""
        return self.traces[-1] if self.traces else None
        
    def get_trace_summary(self, trace: Optional[ReActTrace] = None) -> Dict[str, Any]:
        """Get a summary of a reasoning trace."""
        if trace is None:
            trace = self.get_last_trace()
            
        if not trace:
            return {}
            
        return {
            "question": trace.question,
            "steps": len(trace.steps),
            "success": trace.success,
            "final_answer": trace.final_answer,
            "tools_used": list(set(
                step.tool_name for step in trace.steps 
                if step.step_type == ReActStepType.ACTION and step.tool_name
            )),
            "error": trace.error
        }
        
    def clear_traces(self):
        """Clear all stored reasoning traces."""
        self.traces = []


class ReActAgentWithMemory(ReActAgent):
    """ReAct agent with memory of previous interactions."""
    
    def __init__(self, name: str = "ReAct Agent with Memory", **kwargs):
        super().__init__(name, **kwargs)
        self.memory: List[Dict[str, str]] = []
        
    def process(self, task: str) -> str:
        """Process task with memory context."""
        # Add memory context to the task
        if self.memory:
            memory_context = "\n".join([
                f"Previous Q: {item['question']}\nPrevious A: {item['answer']}"
                for item in self.memory[-3:]  # Last 3 interactions
            ])
            enhanced_task = f"Context from previous interactions:\n{memory_context}\n\nCurrent question: {task}"
        else:
            enhanced_task = task
            
        # Process with enhanced task
        result = super().process(enhanced_task)
        
        # Store in memory
        self.memory.append({
            "question": task,
            "answer": result
        })
        
        return result
        
    def clear_memory(self):
        """Clear the agent's memory."""
        self.memory = []


# Example usage function
def create_react_agent(
    name: str = "ReAct Assistant",
    tools: List[BaseTool] = None,
    llm_config: Optional[LLMConfig] = None,
    enable_reflection: bool = True
) -> ReActAgent:
    """
    Create a ReAct agent with specified configuration.
    
    Args:
        name: Agent name
        tools: List of tools available to the agent
        llm_config: LLM configuration
        enable_reflection: Whether to enable self-reflection
        
    Returns:
        Configured ReAct agent
    """
    # Create LLM router if config provided
    llm_router = None
    if llm_config:
        from ..core.llm_providers import LLMRouter
        llm_router = LLMRouter()
        llm_router.add_provider(llm_config)
        
    # Create agent
    agent = ReActAgent(
        name=name,
        llm_router=llm_router,
        enable_reflection=enable_reflection
    )
    
    # Add tools
    if tools:
        for tool in tools:
            agent.add_tool(tool)
            
    return agent
