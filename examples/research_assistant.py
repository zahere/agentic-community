#!/usr/bin/env python3
"""
Research Assistant Example

Demonstrates how to create a simple research assistant that can
gather information and summarize it.
"""

import os
from agentic_community import SimpleAgent, SearchTool, TextTool


def main():
    # Ensure API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    # Create research assistant
    print("🔍 Research Assistant")
    print("=" * 50)
    print("I can help you research topics and summarize information.")
    print("Note: Search results are simulated in the community edition.\n")
    
    agent = SimpleAgent("ResearchBot")
    agent.add_tool(SearchTool())
    agent.add_tool(TextTool())
    
    # Example research tasks
    research_topics = [
        {
            "topic": "Benefits of meditation",
            "task": "Research the main health benefits of meditation and provide a summary with 3-5 bullet points"
        },
        {
            "topic": "Python vs JavaScript",
            "task": "Compare Python and JavaScript programming languages, focusing on their main use cases and strengths"
        },
        {
            "topic": "Remote work best practices",
            "task": "What are the top 5 best practices for effective remote work?"
        }
    ]
    
    # Run example research tasks
    for research in research_topics:
        print(f"\n📚 Researching: {research['topic']}")
        print(f"📋 Task: {research['task']}")
        print("-" * 50)
        
        result = agent.run(research['task'])
        print(f"\n{result}")
        print("=" * 50)
    
    # Interactive research mode
    print("\n💬 Interactive Research Mode")
    print("Ask me to research any topic (type 'quit' to exit)\n")
    
    while True:
        try:
            query = input("What would you like me to research? ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            # Add context for better results
            if "summarize" not in query.lower() and "summary" not in query.lower():
                query += " Please provide a clear summary."
            
            print("\n🔍 Researching...")
            result = agent.run(query)
            print(f"\n{result}")
            print("\n" + "=" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for using Research Assistant!")


if __name__ == "__main__":
    main()
