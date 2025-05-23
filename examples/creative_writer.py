#!/usr/bin/env python3
"""
Creative Writer Example

Demonstrates how to create a creative writing assistant for
various content creation tasks.
"""

import os
from agentic_community import SimpleAgent, TextTool


def main():
    # Ensure API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    # Create creative writer
    print("✍️  Creative Writing Assistant")
    print("=" * 50)
    print("I'll help you create various types of content.\n")
    
    agent = SimpleAgent("CreativeWriter")
    agent.add_tool(TextTool())
    
    # Writing examples
    writing_tasks = [
        {
            "type": "Product Description",
            "task": """Write a compelling product description for:
            Product: Eco-friendly reusable water bottle
            Features: BPA-free, keeps drinks cold for 24 hours, made from recycled materials
            Target audience: Environmentally conscious consumers"""
        },
        {
            "type": "Email Template",
            "task": """Create a professional email template for:
            Purpose: Following up after a job interview
            Tone: Professional but friendly
            Key points: Thank you, reiterate interest, next steps"""
        },
        {
            "type": "Social Media Post",
            "task": """Write an engaging LinkedIn post about:
            Topic: The importance of continuous learning in tech
            Length: 2-3 paragraphs
            Include: Personal insight and call-to-action"""
        },
        {
            "type": "Blog Introduction",
            "task": """Write an engaging introduction for a blog post about:
            Title: '5 Morning Habits That Boost Productivity'
            Tone: Motivational and practical
            Hook: Start with a relatable scenario"""
        }
    ]
    
    # Menu for writing options
    print("Available writing templates:")
    for i, task in enumerate(writing_tasks, 1):
        print(f"{i}. {task['type']}")
    print(f"{len(writing_tasks) + 1}. Custom writing request")
    print(f"{len(writing_tasks) + 2}. Interactive mode")
    print("0. Exit\n")
    
    while True:
        try:
            choice = input("Select an option (0-6): ")
            
            if choice == '0':
                break
            elif choice.isdigit() and 1 <= int(choice) <= len(writing_tasks):
                # Run selected template
                task = writing_tasks[int(choice) - 1]
                print(f"\n📝 Writing: {task['type']}")
                print("-" * 50)
                result = agent.run(task['task'])
                print(f"\n{result}")
                print("=" * 50)
            elif choice == str(len(writing_tasks) + 1):
                # Custom request
                print("\n📝 Custom Writing Request")
                content_type = input("What type of content? ")
                details = input("Provide details and requirements: ")
                
                print("\n✍️  Writing...")
                result = agent.run(f"Write {content_type}: {details}")
                print(f"\n{result}")
                print("=" * 50)
            elif choice == str(len(writing_tasks) + 2):
                # Interactive mode
                print("\n💬 Interactive Writing Mode")
                print("Describe what you need written (type 'back' to return)\n")
                
                while True:
                    request = input("Writing request: ")
                    if request.lower() in ['back', 'menu']:
                        break
                    
                    print("\n✍️  Writing...")
                    result = agent.run(request)
                    print(f"\n{result}")
                    print("\n" + "-" * 50 + "\n")
                    
                    # Save option
                    save = input("Save this content? (y/n): ")
                    if save.lower() == 'y':
                        filename = input("Filename (without extension): ") + ".txt"
                        with open(filename, 'w') as f:
                            f.write(result)
                        print(f"✅ Saved to {filename}\n")
            else:
                print("Invalid option. Please try again.")
            
            if choice != str(len(writing_tasks) + 2):
                input("\nPress Enter to continue...")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for using Creative Writing Assistant!")


if __name__ == "__main__":
    main()
