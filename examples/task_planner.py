#!/usr/bin/env python3
"""
Task Planner Example

Demonstrates how to create a task planning assistant that helps
organize daily activities and break down complex projects.
"""

import os
from datetime import datetime
from agentic_community import SimpleAgent, TextTool


def main():
    # Ensure API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    # Create task planner
    print("📅 Task Planning Assistant")
    print("=" * 50)
    print("I'll help you organize your tasks and create effective plans.\n")
    
    agent = SimpleAgent("TaskPlanner")
    agent.add_tool(TextTool())
    
    # Example planning scenarios
    planning_examples = [
        {
            "name": "Daily Schedule",
            "request": """Create a daily schedule for a productive workday with:
            - 3 important meetings (9am, 11am, 2pm)
            - 2 hours for deep work
            - Lunch break
            - Email checking times
            - Short breaks between tasks"""
        },
        {
            "name": "Project Breakdown",
            "request": """Break down this project into tasks:
            'Create a personal blog website'
            Include: planning, design, development, content creation, and launch phases"""
        },
        {
            "name": "Weekly Meal Prep",
            "request": """Plan a week of healthy meals with:
            - Breakfast, lunch, and dinner for 5 days
            - Shopping list organized by category
            - Prep time estimates
            - Focus on balanced nutrition"""
        }
    ]
    
    # Run planning examples
    for example in planning_examples:
        print(f"\n📋 Planning: {example['name']}")
        print("-" * 50)
        result = agent.run(example['request'])
        print(result)
        print("=" * 50)
        
        input("\nPress Enter to continue to the next example...")
    
    # Interactive planning mode
    print("\n💬 Interactive Planning Mode")
    print("Describe what you need to plan (type 'quit' to exit)\n")
    
    while True:
        try:
            request = input("What would you like me to help plan? ")
            if request.lower() in ['quit', 'exit', 'q']:
                break
            
            # Get additional context if needed
            if len(request.split()) < 10:
                print("Tell me more details for a better plan...")
                details = input("Additional details: ")
                request = f"{request}. {details}"
            
            print("\n🤔 Creating your plan...")
            result = agent.run(f"Create a detailed plan for: {request}")
            print(f"\n{result}")
            
            # Option to save the plan
            save = input("\nWould you like to save this plan? (y/n): ")
            if save.lower() == 'y':
                filename = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w') as f:
                    f.write(f"Plan created on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Request: {request}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(result)
                print(f"✅ Plan saved to {filename}")
            
            print("\n" + "=" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for using Task Planning Assistant!")


if __name__ == "__main__":
    main()
