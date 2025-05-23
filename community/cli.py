"""
CLI Interface - Community Edition
Command-line interface for Agentic AI Framework
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

import click
import os
import sys
from typing import Optional
from pathlib import Path

from community import SimpleAgent, SearchTool, CalculatorTool, TextTool
from agentic_community.core.licensing import get_license_manager
from agentic_community.core.utils import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


@click.group()
@click.version_option(version="1.0.0", prog_name="Agentic AI Framework")
def cli():
    """Agentic AI Framework - Build autonomous AI agents"""
    pass


@cli.command()
@click.option("--name", prompt="Agent name", help="Name for the agent")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="OpenAI API key")
@click.option("--tools", multiple=True, default=["search", "calculator"], help="Tools to enable")
def create(name: str, api_key: Optional[str], tools):
    """Create a new agent"""
    if not api_key:
        click.echo("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        sys.exit(1)
        
    try:
        # Check license limits
        license_manager = get_license_manager()
        limits = license_manager.get_limits()
        
        click.echo(f"Creating agent '{name}' ({license_manager.get_edition()} edition)...")
        
        # Create agent
        agent = SimpleAgent(name, openai_api_key=api_key)
        
        # Add requested tools
        available_tools = {
            "search": SearchTool,
            "calculator": CalculatorTool,
            "text": TextTool
        }
        
        added_tools = []
        for tool_name in tools[:limits["max_tools"]]:
            if tool_name in available_tools:
                tool = available_tools[tool_name]()
                agent.add_tool(tool)
                added_tools.append(tool_name)
                
        click.echo(f"✅ Agent created with tools: {', '.join(added_tools)}")
        
        if len(tools) > limits["max_tools"]:
            click.echo(f"⚠️  Tool limit reached ({limits['max_tools']}). Upgrade to Enterprise for unlimited tools.")
            
        return agent
        
    except Exception as e:
        click.echo(f"❌ Error creating agent: {e}")
        sys.exit(1)


@cli.command()
@click.option("--name", default="Assistant", help="Agent name")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="OpenAI API key")
def interactive(name: str, api_key: Optional[str]):
    """Start interactive session with an agent"""
    if not api_key:
        click.echo("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        sys.exit(1)
        
    # Create agent with all tools
    agent = SimpleAgent(name, openai_api_key=api_key)
    agent.add_tool(SearchTool())
    agent.add_tool(CalculatorTool())
    agent.add_tool(TextTool())
    
    license_manager = get_license_manager()
    
    click.echo(f"\n🤖 Agentic AI Framework - {license_manager.get_edition().title()} Edition")
    click.echo(f"Agent '{name}' is ready! Type 'exit' to quit.\n")
    
    while True:
        try:
            # Get user input
            task = click.prompt("You", type=str)
            
            if task.lower() in ["exit", "quit", "bye"]:
                click.echo("\n👋 Goodbye!")
                break
                
            # Execute task
            click.echo(f"\n{name}: Thinking...")
            result = agent.run(task)
            
            # Display result
            click.echo(f"\n{name}: {result}\n")
            
        except KeyboardInterrupt:
            click.echo("\n\n👋 Goodbye!")
            break
        except Exception as e:
            click.echo(f"\n❌ Error: {e}\n")


@cli.command()
@click.argument("task")
@click.option("--name", default="Assistant", help="Agent name")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="OpenAI API key")
@click.option("--tools", multiple=True, default=["search", "calculator"], help="Tools to use")
def run(task: str, name: str, api_key: Optional[str], tools):
    """Run a single task"""
    if not api_key:
        click.echo("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        sys.exit(1)
        
    try:
        # Create agent
        agent = SimpleAgent(name, openai_api_key=api_key)
        
        # Add tools
        available_tools = {
            "search": SearchTool,
            "calculator": CalculatorTool,
            "text": TextTool
        }
        
        for tool_name in tools:
            if tool_name in available_tools:
                tool = available_tools[tool_name]()
                agent.add_tool(tool)
                
        # Execute task
        click.echo(f"\n🤖 Running task: {task}")
        result = agent.run(task)
        
        # Display result
        click.echo(f"\n📋 Result:\n{result}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)


@cli.command()
def info():
    """Show framework information"""
    license_manager = get_license_manager()
    
    click.echo("\n🤖 Agentic AI Framework")
    click.echo(f"Version: 1.0.0")
    click.echo(f"Edition: {license_manager.get_edition().title()}")
    
    click.echo("\n📋 Features:")
    features = {
        "Basic Reasoning": license_manager.check_feature("basic_reasoning"),
        "Advanced Reasoning": license_manager.check_feature("advanced_reasoning"),
        "Self-Reflection": license_manager.check_feature("self_reflection"),
        "Multi-Agent": license_manager.check_feature("multi_agent"),
        "All LLM Providers": license_manager.check_feature("all_llm_providers")
    }
    
    for feature, available in features.items():
        status = "✅" if available else "❌"
        click.echo(f"  {status} {feature}")
        
    limits = license_manager.get_limits()
    click.echo("\n🔢 Limits:")
    click.echo(f"  Max Agents: {limits['max_agents'] or 'Unlimited'}")
    click.echo(f"  Max Tools: {limits['max_tools'] or 'Unlimited'}")
    click.echo(f"  Max Iterations: {limits['max_iterations']}")
    click.echo(f"  LLM Providers: {', '.join(limits['llm_providers'])}")
    click.echo(f"  Support Level: {limits['support_level'].title()}")
    
    if license_manager.get_edition() == "community":
        click.echo("\n💡 Upgrade to Enterprise for advanced features:")
        click.echo("   https://agentic-ai.com/enterprise")


@cli.command()
@click.option("--host", default="0.0.0.0", help="API host")
@click.option("--port", default=8000, help="API port")
def serve(host: str, port: int):
    """Start the API server"""
    click.echo(f"\n🚀 Starting API server on {host}:{port}")
    click.echo("Press CTRL+C to stop\n")
    
    try:
        from agentic_community.api import main as api_main
        os.environ["API_HOST"] = host
        os.environ["API_PORT"] = str(port)
        api_main()
    except ImportError:
        click.echo("❌ API dependencies not installed. Install with: pip install agentic-community[api]")
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\n\n✅ API server stopped")


@cli.command()
@click.option("--key", help="License key to activate")
def license(key: Optional[str]):
    """Manage license"""
    license_manager = get_license_manager()
    
    if key:
        # Activate license
        click.echo(f"Activating license...")
        if license_manager.validate_license(key):
            # Save license
            license_path = Path.home() / ".agentic" / "license.json"
            license_manager.save_license(license_path)
            click.echo("✅ License activated successfully!")
            click.echo(f"Edition: {license_manager.get_edition().title()}")
        else:
            click.echo("❌ Invalid license key")
    else:
        # Show current license
        click.echo(f"\n📄 Current License:")
        click.echo(f"Edition: {license_manager.get_edition().title()}")
        
        if license_manager.license:
            click.echo(f"Company: {license_manager.license.company or 'N/A'}")
            click.echo(f"Seats: {license_manager.license.seats}")
            if license_manager.license.expires_at:
                click.echo(f"Expires: {license_manager.license.expires_at.date()}")


def main():
    """Main CLI entry point"""
    cli()


if __name__ == "__main__":
    main()
