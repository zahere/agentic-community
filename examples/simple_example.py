"""
Simple Example - Community Edition
Shows basic agent usage with tools
"""

from agentic_community import SimpleAgent, SearchTool, CalculatorTool, TextTool


def main():
    """Run a simple example with the lightweight community edition."""
    
    print("🤖 Agentic Community Edition - Simple Example\n")
    
    # Create tools
    print("Creating tools...")
    search_tool = SearchTool()
    calc_tool = CalculatorTool()
    text_tool = TextTool()
    
    # Create agent with tools
    print("Creating Simple Agent...")
    agent = SimpleAgent("Assistant", tools=[search_tool, calc_tool, text_tool])
    
    print(f"Agent initialized with {len(agent.tools)} tools\n")
    
    # Example 1: Simple calculation task
    print("="*50)
    print("Example 1: Calculation Task")
    print("="*50)
    
    task1 = "Calculate the sum of 145 and 387"
    print(f"Task: {task1}")
    result1 = agent.execute(task1)
    print(f"Result:\n{result1}\n")
    
    # Example 2: Search task
    print("="*50)
    print("Example 2: Search Task")
    print("="*50)
    
    task2 = "Search for information about sustainable energy"
    print(f"Task: {task2}")
    result2 = agent.execute(task2)
    print(f"Result:\n{result2}\n")
    
    # Example 3: Text analysis
    print("="*50)
    print("Example 3: Text Analysis Task")
    print("="*50)
    
    task3 = "Analyze this text: The future of AI lies in making it more accessible and ethical"
    print(f"Task: {task3}")
    result3 = agent.execute(task3)
    print(f"Result:\n{result3}\n")
    
    # Example 4: Combined task
    print("="*50)
    print("Example 4: Multi-step Task")
    print("="*50)
    
    task4 = "I have a budget of $2000 for a 5-day trip. Calculate daily budget and search for budget travel tips"
    print(f"Task: {task4}")
    result4 = agent.execute(task4)
    print(f"Result:\n{result4}\n")
    
    # Show agent state
    print("="*50)
    print("Agent State")
    print("="*50)
    state = agent.get_state()
    print(f"Agent: {state['name']}")
    print(f"Type: {state['agent_type']}")
    print(f"Tools: {state['tool_count']}")
    print(f"History: {state['history_length']} tasks executed")
    
    # Show features
    print("\n" + "="*50)
    print("Community Edition Features:")
    print("="*50)
    print("✓ Lightweight - no external dependencies")
    print("✓ Simple sequential reasoning")
    print("✓ Basic tool integration")
    print("✓ State management and history")
    print("✓ Easy to extend and customize")
    
    print("\n" + "="*50)
    print("Limitations:")
    print("="*50)
    print("• Basic reasoning only (no advanced strategies)")
    print("• No self-reflection or learning")
    print("• Single agent execution")
    print("• Limited tool capabilities")
    
    print("\n🚀 For advanced features like self-reflection, multi-agent orchestration,")
    print("   and enterprise tools, upgrade to the Enterprise Edition!")


if __name__ == "__main__":
    main()
