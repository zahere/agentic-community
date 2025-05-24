"""
Reasoning Agent for Community Edition

This agent implements basic reasoning capabilities including step-by-step
thinking and simple chain-of-thought reasoning.
"""

import asyncio
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
import json

from agentic_community.core.base import BaseAgent
from agentic_community.core.llm_providers import create_llm_client


class ReasoningType(Enum):
    """Types of reasoning approaches"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"


@dataclass
class ReasoningStep:
    """Represents a single step in reasoning"""
    step_number: int
    description: str
    reasoning_type: ReasoningType
    conclusion: str
    confidence: float
    supporting_evidence: List[str]


@dataclass
class ReasoningChain:
    """Represents a complete reasoning chain"""
    problem: str
    steps: List[ReasoningStep]
    final_conclusion: str
    overall_confidence: float
    assumptions: List[str]


class ReasoningAgent(BaseAgent):
    """
    Agent that can perform structured reasoning on problems.
    
    Features:
    - Step-by-step reasoning
    - Multiple reasoning strategies
    - Chain-of-thought prompting
    - Confidence scoring
    - Assumption tracking
    """
    
    def __init__(self, name: str, llm_provider: str = "openai", **kwargs):
        super().__init__(name)
        self.llm_client = create_llm_client(llm_provider, **kwargs)
        self.reasoning_history = []
        
    async def reason(
        self,
        problem: str,
        reasoning_type: Optional[ReasoningType] = None,
        max_steps: int = 5,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningChain:
        """
        Perform structured reasoning on a problem.
        
        Args:
            problem: The problem to reason about
            reasoning_type: Specific reasoning type to use (or auto-select)
            max_steps: Maximum reasoning steps
            context: Additional context
            
        Returns:
            Complete reasoning chain with conclusion
        """
        # Determine best reasoning type if not specified
        if not reasoning_type:
            reasoning_type = await self._select_reasoning_type(problem)
        
        # Extract assumptions
        assumptions = await self._identify_assumptions(problem, context)
        
        # Perform step-by-step reasoning
        steps = await self._perform_reasoning(
            problem,
            reasoning_type,
            max_steps,
            context
        )
        
        # Generate final conclusion
        final_conclusion = await self._synthesize_conclusion(
            problem,
            steps,
            assumptions
        )
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(steps)
        
        chain = ReasoningChain(
            problem=problem,
            steps=steps,
            final_conclusion=final_conclusion,
            overall_confidence=overall_confidence,
            assumptions=assumptions
        )
        
        self.reasoning_history.append(chain)
        return chain
    
    async def _select_reasoning_type(self, problem: str) -> ReasoningType:
        """
        Automatically select the best reasoning type for a problem.
        """
        messages = [
            {"role": "system", "content": "You are an expert at selecting reasoning strategies."},
            {"role": "user", "content": f"""
            For this problem, select the best reasoning approach:
            
            Problem: {problem}
            
            Options:
            - DEDUCTIVE: From general principles to specific conclusions
            - INDUCTIVE: From specific observations to general principles
            - ABDUCTIVE: Finding the best explanation for observations
            - ANALOGICAL: Reasoning by comparison and similarity
            - CAUSAL: Understanding cause and effect relationships
            
            Respond with only the reasoning type name.
            """}
        ]
        
        response = await self.llm_client.complete(messages)
        
        # Parse response and return appropriate type
        response_upper = response.content.upper().strip()
        for reasoning_type in ReasoningType:
            if reasoning_type.value.upper() in response_upper:
                return reasoning_type
        
        # Default to deductive
        return ReasoningType.DEDUCTIVE
    
    async def _identify_assumptions(self, problem: str, context: Optional[Dict[str, Any]]) -> List[str]:
        """
        Identify underlying assumptions in the problem.
        """
        prompt = f"""
        Identify key assumptions in this problem:
        
        Problem: {problem}
        """
        
        if context:
            prompt += f"\nContext: {json.dumps(context)}"
        
        prompt += "\n\nList assumptions, one per line."
        
        messages = [
            {"role": "system", "content": "You are an expert at identifying assumptions."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.llm_client.complete(messages)
        assumptions = [line.strip() for line in response.content.split('\n') if line.strip()]
        return assumptions
    
    async def _perform_reasoning(
        self,
        problem: str,
        reasoning_type: ReasoningType,
        max_steps: int,
        context: Optional[Dict[str, Any]]
    ) -> List[ReasoningStep]:
        """
        Perform step-by-step reasoning.
        """
        steps = []
        current_state = problem
        
        for step_num in range(1, max_steps + 1):
            # Generate next reasoning step
            step = await self._generate_reasoning_step(
                current_state,
                reasoning_type,
                step_num,
                steps,
                context
            )
            
            steps.append(step)
            
            # Check if we've reached a conclusion
            if await self._is_conclusion_reached(step):
                break
            
            # Update current state
            current_state = f"{problem}\n\nCurrent conclusion: {step.conclusion}"
        
        return steps
    
    async def _generate_reasoning_step(
        self,
        current_state: str,
        reasoning_type: ReasoningType,
        step_number: int,
        previous_steps: List[ReasoningStep],
        context: Optional[Dict[str, Any]]
    ) -> ReasoningStep:
        """
        Generate a single reasoning step.
        """
        # Build prompt based on reasoning type
        prompt = self._build_step_prompt(
            current_state,
            reasoning_type,
            step_number,
            previous_steps
        )
        
        messages = [
            {"role": "system", "content": f"You are using {reasoning_type.value} reasoning."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.llm_client.complete(messages)
        
        # Parse response into reasoning step
        return await self._parse_reasoning_step(
            response.content,
            step_number,
            reasoning_type
        )
    
    def _build_step_prompt(self, current_state: str, reasoning_type: ReasoningType, 
                          step_number: int, previous_steps: List[ReasoningStep]) -> str:
        """
        Build prompt for generating a reasoning step.
        """
        prompt = f"""Step {step_number} of {reasoning_type.value} reasoning:
        
        Current state: {current_state}
        """
        
        if previous_steps:
            prompt += "\n\nPrevious steps:"
            for step in previous_steps:
                prompt += f"\n{step.step_number}. {step.conclusion}"
        
        prompt += f"""
        
        Provide the next reasoning step in this format:
        DESCRIPTION: [What you're analyzing]
        EVIDENCE: [Supporting facts or observations]
        CONCLUSION: [What you conclude from this step]
        CONFIDENCE: [0-100% confidence in this conclusion]
        """
        
        return prompt
    
    async def _parse_reasoning_step(
        self,
        response: str,
        step_number: int,
        reasoning_type: ReasoningType
    ) -> ReasoningStep:
        """
        Parse LLM response into a reasoning step.
        """
        # Simple parsing - could be enhanced
        lines = response.strip().split('\n')
        
        description = ""
        evidence = []
        conclusion = ""
        confidence = 0.5
        
        for line in lines:
            if line.startswith("DESCRIPTION:"):
                description = line.replace("DESCRIPTION:", "").strip()
            elif line.startswith("EVIDENCE:"):
                evidence = [line.replace("EVIDENCE:", "").strip()]
            elif line.startswith("CONCLUSION:"):
                conclusion = line.replace("CONCLUSION:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.replace("CONFIDENCE:", "").strip().rstrip('%')
                    confidence = float(conf_str) / 100
                except:
                    confidence = 0.5
        
        return ReasoningStep(
            step_number=step_number,
            description=description or "Reasoning step",
            reasoning_type=reasoning_type,
            conclusion=conclusion or "No conclusion",
            confidence=confidence,
            supporting_evidence=evidence
        )
    
    async def _is_conclusion_reached(self, step: ReasoningStep) -> bool:
        """
        Check if a reasoning step represents a final conclusion.
        """
        conclusion_keywords = [
            "therefore", "thus", "in conclusion", "finally",
            "the answer is", "we can conclude", "this proves"
        ]
        
        conclusion_lower = step.conclusion.lower()
        return any(keyword in conclusion_lower for keyword in conclusion_keywords)
    
    async def _synthesize_conclusion(
        self,
        problem: str,
        steps: List[ReasoningStep],
        assumptions: List[str]
    ) -> str:
        """
        Synthesize all reasoning steps into a final conclusion.
        """
        steps_summary = "\n".join([
            f"{step.step_number}. {step.conclusion} (confidence: {step.confidence:.0%})"
            for step in steps
        ])
        
        messages = [
            {"role": "system", "content": "You are synthesizing a reasoning chain into a final conclusion."},
            {"role": "user", "content": f"""
            Problem: {problem}
            
            Assumptions:
            {chr(10).join(f'- {a}' for a in assumptions)}
            
            Reasoning steps:
            {steps_summary}
            
            Provide a clear, concise final conclusion that addresses the original problem.
            """}
        ]
        
        response = await self.llm_client.complete(messages)
        return response.content
    
    def _calculate_overall_confidence(self, steps: List[ReasoningStep]) -> float:
        """
        Calculate overall confidence from individual steps.
        """
        if not steps:
            return 0.0
        
        # Weighted average, giving more weight to later steps
        total_weight = 0
        weighted_sum = 0
        
        for i, step in enumerate(steps):
            weight = i + 1  # Later steps have more weight
            weighted_sum += step.confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    async def solve_problem(
        self,
        problem: str,
        approach: str = "step_by_step"
    ) -> Dict[str, Any]:
        """
        Solve a problem using specified approach.
        
        Args:
            problem: Problem statement
            approach: "step_by_step", "backwards", or "decompose"
            
        Returns:
            Solution with reasoning trace
        """
        if approach == "step_by_step":
            return await self._solve_step_by_step(problem)
        elif approach == "backwards":
            return await self._solve_backwards(problem)
        elif approach == "decompose":
            return await self._solve_by_decomposition(problem)
        else:
            raise ValueError(f"Unknown approach: {approach}")
    
    async def _solve_step_by_step(self, problem: str) -> Dict[str, Any]:
        """
        Solve problem with forward chaining.
        """
        chain = await self.reason(problem, reasoning_type=ReasoningType.DEDUCTIVE)
        
        return {
            "approach": "step_by_step",
            "problem": problem,
            "solution": chain.final_conclusion,
            "steps": [
                {
                    "number": step.step_number,
                    "description": step.description,
                    "conclusion": step.conclusion
                }
                for step in chain.steps
            ],
            "confidence": chain.overall_confidence
        }
    
    async def _solve_backwards(self, problem: str) -> Dict[str, Any]:
        """
        Solve problem with backward chaining.
        """
        messages = [
            {"role": "system", "content": "You solve problems by working backwards from the goal."},
            {"role": "user", "content": f"""
            Problem: {problem}
            
            Work backwards from the desired solution. What would need to be true?
            Provide your reasoning in reverse order.
            """}
        ]
        
        response = await self.llm_client.complete(messages)
        
        return {
            "approach": "backwards",
            "problem": problem,
            "solution": response.content,
            "confidence": 0.7  # Default confidence
        }
    
    async def _solve_by_decomposition(self, problem: str) -> Dict[str, Any]:
        """
        Solve problem by decomposing into subproblems.
        """
        # Decompose the problem
        messages = [
            {"role": "system", "content": "You decompose complex problems into simpler subproblems."},
            {"role": "user", "content": f"""
            Problem: {problem}
            
            Break this down into smaller, manageable subproblems.
            List each subproblem on a new line.
            """}
        ]
        
        response = await self.llm_client.complete(messages)
        subproblems = [line.strip() for line in response.content.split('\n') if line.strip()]
        
        # Solve each subproblem
        solutions = []
        for subproblem in subproblems:
            sub_solution = await self._solve_subproblem(subproblem)
            solutions.append({
                "subproblem": subproblem,
                "solution": sub_solution
            })
        
        # Combine solutions
        combined = await self._combine_solutions(problem, solutions)
        
        return {
            "approach": "decomposition",
            "problem": problem,
            "subproblems": solutions,
            "solution": combined,
            "confidence": 0.8
        }
    
    async def _solve_subproblem(self, subproblem: str) -> str:
        """
        Solve a single subproblem.
        """
        messages = [
            {"role": "system", "content": "You solve focused subproblems concisely."},
            {"role": "user", "content": f"Solve: {subproblem}"}
        ]
        
        response = await self.llm_client.complete(messages)
        return response.content
    
    async def _combine_solutions(self, problem: str, solutions: List[Dict[str, str]]) -> str:
        """
        Combine subproblem solutions into final solution.
        """
        solutions_text = "\n".join([
            f"Subproblem: {s['subproblem']}\nSolution: {s['solution']}\n"
            for s in solutions
        ])
        
        messages = [
            {"role": "system", "content": "You combine subproblem solutions into comprehensive solutions."},
            {"role": "user", "content": f"""
            Original problem: {problem}
            
            Subproblem solutions:
            {solutions_text}
            
            Combine these into a complete solution to the original problem.
            """}
        ]
        
        response = await self.llm_client.complete(messages)
        return response.content
    
    async def process(self, input_data: str) -> str:
        """
        Process input using reasoning.
        """
        # Perform reasoning and return conclusion
        chain = await self.reason(input_data)
        return chain.final_conclusion


# Example usage
if __name__ == "__main__":
    async def demo_reasoning():
        # Create reasoning agent
        reasoner = ReasoningAgent("Aristotle", llm_provider="openai")
        
        # Test basic reasoning
        print("=== Basic Reasoning Demo ===")
        problem = "If all humans are mortal, and Socrates is human, what can we conclude?"
        
        chain = await reasoner.reason(problem)
        
        print(f"Problem: {problem}")
        print(f"\nReasoning type: {chain.steps[0].reasoning_type.value}")
        print(f"\nSteps:")
        for step in chain.steps:
            print(f"{step.step_number}. {step.conclusion} (confidence: {step.confidence:.0%})")
        
        print(f"\nFinal conclusion: {chain.final_conclusion}")
        print(f"Overall confidence: {chain.overall_confidence:.0%}")
        
        # Test problem solving approaches
        print("\n=== Problem Solving Demo ===")
        math_problem = "A train travels 60 miles in 1.5 hours. What is its average speed?"
        
        # Step by step
        solution1 = await reasoner.solve_problem(math_problem, "step_by_step")
        print(f"\nStep-by-step solution: {solution1['solution']}")
        
        # Decomposition
        solution2 = await reasoner.solve_problem(math_problem, "decompose")
        print(f"\nDecomposition solution: {solution2['solution']}")
    
    # Run the demo
    asyncio.run(demo_reasoning())
