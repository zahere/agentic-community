#!/usr/bin/env python3
"""
Calculator Bot Example

A simple math assistant that can handle various calculations.
"""

import os
from agentic_community import SimpleAgent, CalculatorTool, TextTool


def main():
    # Ensure API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    # Create calculator agent
    print("🧮 Calculator Bot")
    print("=" * 40)
    
    agent = SimpleAgent("MathBot")
    agent.add_tool(CalculatorTool())
    agent.add_tool(TextTool())  # For formatting results
    
    # Example calculations
    examples = [
        "What is 15% tip on a $85.50 bill?",
        "Calculate the area of a circle with radius 7.5 meters",
        "If I save $250 per month for 2 years, how much will I have?",
        "What's the square root of 144?",
        "Convert 98.6°F to Celsius using the formula (F-32)*5/9"
    ]
    
    for example in examples:
        print(f"\n📝 Task: {example}")
        result = agent.run(example)
        print(f"📊 Result: {result}")
        print("-" * 40)
    
    # Interactive mode
    print("\n💬 Interactive Mode (type 'quit' to exit)")
    while True:
        try:
            task = input("\nEnter calculation: ")
            if task.lower() in ['quit', 'exit', 'q']:
                break
            
            result = agent.run(task)
            print(f"Result: {result}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for using Calculator Bot!")


if __name__ == "__main__":
    main()
