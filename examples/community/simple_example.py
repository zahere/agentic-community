"""
Simple Example - Community Edition
Shows basic agent usage with tools
"""

import os
from dotenv import load_dotenv

from community import SimpleAgent, SearchTool, CalculatorTool, TextTool


def main():
    """Run a simple example."""
    # Load environment variables
    load_dotenv()
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY in your environment")
        return
        
    # Create agent
    print("Creating Simple Agent...")
    agent = SimpleAgent("Assistant", openai_api_key=api_key)
    
    # Add tools (community edition limited to 3 tools)
    print("Adding tools...")
    agent.add_tool(SearchTool())
    agent.add_tool(CalculatorTool())
    agent.add_tool(TextTool())
    
    # Example 1: Simple task
    print("\n" + "="*50)
    print("Example 1: Simple Planning Task")
    print("="*50)
    
    task1 = "Help me plan a weekend trip to Paris"
    result1 = agent.run(task1)
    print(f"\nTask: {task1}")
    print(f"Result:\n{result1}")
    
    # Example 2: Task with calculation
    print("\n" + "="*50)
    print("Example 2: Task with Calculation")
    print("="*50)
    
    task2 = "I have a budget of $2000 for a 5-day trip. How much can I spend per day?"
    result2 = agent.run(task2)
    print(f"\nTask: {task2}")
    print(f"Result:\n{result2}")
    
    # Example 3: Text processing
    print("\n" + "="*50)
    print("Example 3: Text Processing Task")
    print("="*50)
    
    task3 = "Summarize the key points about sustainable travel"
    result3 = agent.run(task3)
    print(f"\nTask: {task3}")
    print(f"Result:\n{result3}")
    
    # Show limitations
    print("\n" + "="*50)
    print("Community Edition Limitations:")
    print("="*50)
    print("- Basic sequential reasoning only")
    print("- Limited to 3 tools")
    print("- No self-reflection or revision")
    print("- Single agent execution")
    print("- OpenAI API only")
    print("\nFor advanced features, consider upgrading to Enterprise Edition")


if __name__ == "__main__":
    main()
