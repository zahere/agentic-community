#!/usr/bin/env python3
"""
Reasoning Agent Demo - Agentic Community Edition

This example demonstrates the ReasoningAgent's ability to perform various types
of structured reasoning including deductive, inductive, abductive, analogical,
and causal reasoning with confidence scoring.
"""

import asyncio
import os
from agentic_community.agents.reasoning_agent import ReasoningAgent
from agentic_community.tools.calculator_tool import CalculatorTool
from agentic_community.tools.search_tool import SearchTool


async def deductive_reasoning_demo():
    """Demonstrate deductive reasoning capabilities."""
    print("=== Deductive Reasoning Demo ===\n")
    
    # Create reasoning agent
    reasoner = ReasoningAgent(
        name="LogicalReasoner",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )
    
    # Add calculator tool for logical operations
    reasoner.add_tool(CalculatorTool())
    
    # Deductive reasoning problem
    premises = [
        "All software engineers know how to code",
        "Sarah is a software engineer",
        "People who know how to code can build applications"
    ]
    
    result = await reasoner.reason(
        query="What can we conclude about Sarah's abilities?",
        reasoning_type="deductive",
        context={"premises": premises}
    )
    
    print("Premises:")
    for p in premises:
        print(f"  - {p}")
    print(f"\nConclusion: {result['conclusion']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Reasoning Steps: {len(result['steps'])}")


async def inductive_reasoning_demo():
    """Demonstrate inductive reasoning from observations."""
    print("\n\n=== Inductive Reasoning Demo ===\n")
    
    # Create reasoning agent
    reasoner = ReasoningAgent(
        name="InductiveReasoner",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )
    
    # Inductive reasoning from observations
    observations = [
        "Customer A bought the product after seeing 3 reviews",
        "Customer B bought the product after seeing 5 reviews",
        "Customer C bought the product after seeing 4 reviews",
        "Customer D did not buy after seeing 1 review",
        "Customer E bought the product after seeing 6 reviews"
    ]
    
    result = await reasoner.reason(
        query="What pattern can we identify about customer purchasing behavior?",
        reasoning_type="inductive",
        context={"observations": observations}
    )
    
    print("Observations:")
    for obs in observations:
        print(f"  - {obs}")
    print(f"\nPattern Identified: {result['conclusion']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")


async def problem_solving_demo():
    """Demonstrate complex problem solving with step-by-step reasoning."""
    print("\n\n=== Problem Solving Demo ===\n")
    
    # Create reasoning agent with tools
    problem_solver = ReasoningAgent(
        name="ProblemSolver",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )
    
    # Add tools for problem solving
    problem_solver.add_tool(CalculatorTool())
    problem_solver.add_tool(SearchTool())
    
    # Complex problem
    problem = """
    A company needs to optimize its delivery routes. They have:
    - 5 delivery trucks
    - 50 packages to deliver
    - Each truck can carry 15 packages
    - Average delivery time is 10 minutes per package
    - Trucks operate for 8 hours per day
    
    How should they organize the deliveries to maximize efficiency?
    """
    
    result = await problem_solver.solve_problem(
        problem=problem,
        approach="systematic",
        max_steps=5
    )
    
    print(f"Problem: {problem}")
    print("\nSolution Process:")
    for i, step in enumerate(result['steps'], 1):
        print(f"\nStep {i}: {step['description']}")
        print(f"Action: {step['action']}")
        if 'result' in step:
            print(f"Result: {step['result']}")
    
    print(f"\nFinal Solution: {result['solution']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")


async def causal_reasoning_demo():
    """Demonstrate causal reasoning and analysis."""
    print("\n\n=== Causal Reasoning Demo ===\n")
    
    # Create reasoning agent
    causal_reasoner = ReasoningAgent(
        name="CausalAnalyst",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )
    
    # Causal analysis scenario
    scenario = {
        "event": "Website traffic decreased by 40% last week",
        "timeline": [
            "Monday: Deployed new UI update",
            "Tuesday: Page load time increased to 5 seconds",
            "Wednesday: Customer complaints about slow loading",
            "Thursday: Traffic started declining",
            "Friday: 40% decrease in traffic observed"
        ],
        "additional_info": [
            "No marketing campaign changes",
            "Competitor launched new feature on Wednesday",
            "Server logs show increased response times"
        ]
    }
    
    result = await causal_reasoner.reason(
        query="What caused the traffic decrease?",
        reasoning_type="causal",
        context=scenario
    )
    
    print(f"Event: {scenario['event']}")
    print("\nTimeline:")
    for event in scenario['timeline']:
        print(f"  - {event}")
    print("\nCausal Analysis:")
    print(f"Primary Cause: {result['conclusion']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print("\nReasoning Chain:")
    for i, step in enumerate(result['steps'], 1):
        print(f"  {i}. {step}")


async def confidence_scoring_demo():
    """Demonstrate reasoning with confidence scoring."""
    print("\n\n=== Confidence Scoring Demo ===\n")
    
    # Create reasoning agent
    reasoner = ReasoningAgent(
        name="ConfidenceScorer",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )
    
    # Multiple scenarios with varying certainty
    scenarios = [
        {
            "query": "Will it rain tomorrow?",
            "context": {
                "current_weather": "Cloudy with 80% humidity",
                "forecast": "70% chance of precipitation",
                "season": "Rainy season"
            }
        },
        {
            "query": "Is this email a phishing attempt?",
            "context": {
                "sender": "security@bankofamerica.co",
                "content": "Click here to verify your account",
                "grammar": "Multiple spelling errors",
                "urgency": "Action required within 24 hours"
            }
        },
        {
            "query": "Should we invest in this startup?",
            "context": {
                "team": "Experienced founders with 2 successful exits",
                "market": "Growing at 30% annually",
                "competition": "3 established players",
                "funding": "Seed round, seeking $2M"
            }
        }
    ]
    
    print("Analyzing multiple scenarios with confidence scoring:\n")
    
    for scenario in scenarios:
        result = await reasoner.reason(
            query=scenario["query"],
            reasoning_type="analytical",
            context=scenario["context"]
        )
        
        print(f"Question: {scenario['query']}")
        print(f"Conclusion: {result['conclusion']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"Key Factors: {', '.join(result.get('key_factors', []))}")
        print("-" * 50)


async def main():
    """Run all reasoning agent demos."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Run deductive reasoning demo
        await deductive_reasoning_demo()
        
        # Run inductive reasoning demo
        await inductive_reasoning_demo()
        
        # Run problem solving demo
        await problem_solving_demo()
        
        # Run causal reasoning demo
        await causal_reasoning_demo()
        
        # Run confidence scoring demo
        await confidence_scoring_demo()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())