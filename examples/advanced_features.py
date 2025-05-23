"""
Example demonstrating advanced features: web scraping and performance benchmarking.

This example shows how to:
1. Use the web scraper tool to extract content
2. Benchmark agent performance
3. Compare different tool configurations
"""

import asyncio
import os
from typing import List

from agentic_community import SimpleAgent
from agentic_community.tools import (
    SearchTool,
    WebScraperTool,
    TextTool,
    DataFrameTool
)
from agentic_community.core.benchmarks import PerformanceBenchmark


async def demo_web_scraping():
    """Demonstrate web scraping capabilities."""
    print("=== Web Scraping Demo ===\n")
    
    # Create agent with web scraping tool
    agent = SimpleAgent("WebResearcher")
    agent.add_tool(WebScraperTool())
    agent.add_tool(TextTool())
    
    # Example 1: Scrape a single page
    print("1. Scraping a single page...")
    result = await agent.run(
        "Use the web scraper to extract the main content from https://example.com"
    )
    print(f"Result: {result[:500]}...\n")
    
    # Example 2: Extract specific elements
    print("2. Extracting specific elements...")
    result = await agent.run(
        "Use the web scraper to extract all links and images from https://example.com"
    )
    print(f"Result: {result[:500]}...\n")
    
    # Example 3: Scrape and summarize
    print("3. Scraping and summarizing...")
    result = await agent.run(
        "Scrape https://example.com and use the text tool to summarize the main content"
    )
    print(f"Result: {result}\n")


async def demo_performance_benchmarking():
    """Demonstrate performance benchmarking."""
    print("\n=== Performance Benchmarking Demo ===\n")
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark()
    
    # Test different agent configurations
    agents = []
    
    # Agent 1: Basic configuration
    agent1 = SimpleAgent("BasicAgent")
    agent1.add_tool(SearchTool())
    agents.append(("Basic", agent1))
    
    # Agent 2: With multiple tools
    agent2 = SimpleAgent("AdvancedAgent")
    agent2.add_tool(SearchTool())
    agent2.add_tool(WebScraperTool())
    agent2.add_tool(TextTool())
    agents.append(("Advanced", agent2))
    
    # Test tasks
    test_tasks = [
        "What is the weather today?",
        "Search for information about Python programming",
        "Find and summarize recent news about AI"
    ]
    
    # Benchmark each agent
    summaries = []
    for name, agent in agents:
        print(f"Benchmarking {name} agent...")
        summary = await benchmark.benchmark_agent(
            agent,
            test_tasks,
            name=name,
            iterations=3
        )
        summaries.append(summary)
        
        print(f"  Average duration: {summary.avg_duration:.3f}s")
        print(f"  Memory usage: {summary.avg_memory:.1f}MB")
        print(f"  Success rate: {summary.success_rate:.1%}\n")
    
    # Compare results
    print("Generating comparison report...")
    report = benchmark.generate_report(summaries, "agent_comparison.md")
    print("Report saved to benchmarks/agent_comparison.md")
    
    # Save raw results
    benchmark.save_results("benchmark_results.json")
    print("Raw results saved to benchmarks/benchmark_results.json")


async def demo_tool_benchmarking():
    """Benchmark individual tools."""
    print("\n=== Tool Benchmarking Demo ===\n")
    
    benchmark = PerformanceBenchmark()
    
    # Benchmark search tool
    print("Benchmarking SearchTool...")
    search_tool = SearchTool()
    search_inputs = [
        {"query": "Python programming"},
        {"query": "Machine learning algorithms"},
        {"query": "Web development frameworks"}
    ]
    
    search_summary = await benchmark.benchmark_tool(
        search_tool,
        search_inputs,
        name="SearchTool",
        iterations=5
    )
    
    print(f"  Average duration: {search_summary.avg_duration:.3f}s")
    print(f"  Success rate: {search_summary.success_rate:.1%}\n")
    
    # Benchmark web scraper
    print("Benchmarking WebScraperTool...")
    scraper_tool = WebScraperTool()
    scraper_inputs = [
        {
            "url": "https://example.com",
            "extract_text": True,
            "extract_links": False
        }
    ]
    
    scraper_summary = await benchmark.benchmark_tool(
        scraper_tool,
        scraper_inputs,
        name="WebScraperTool",
        iterations=3
    )
    
    print(f"  Average duration: {scraper_summary.avg_duration:.3f}s")
    print(f"  Success rate: {scraper_summary.success_rate:.1%}\n")
    
    # Compare tools
    comparison = benchmark.compare_benchmarks([search_summary, scraper_summary])
    print(f"Fastest tool: {comparison['fastest'].name}")
    print(f"Most memory efficient: {comparison['least_memory'].name}")


async def demo_advanced_agent():
    """Demonstrate an advanced agent with multiple capabilities."""
    print("\n=== Advanced Agent Demo ===\n")
    
    # Create a research agent with multiple tools
    agent = SimpleAgent("ResearchAssistant")
    agent.add_tool(SearchTool())
    agent.add_tool(WebScraperTool())
    agent.add_tool(DataFrameTool())
    
    # Complex research task
    task = """
    Research the topic of 'sustainable energy solutions':
    1. Search for recent developments
    2. Find and scrape a relevant article
    3. Create a summary with key statistics
    """
    
    print("Executing complex research task...")
    start_time = asyncio.get_event_loop().time()
    
    result = await agent.run(task)
    
    duration = asyncio.get_event_loop().time() - start_time
    print(f"\nTask completed in {duration:.2f} seconds")
    print(f"Result:\n{result}")


async def main():
    """Run all demonstrations."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Note: Set OPENAI_API_KEY for full functionality")
        print("Using mock mode for demonstration\n")
    
    try:
        # Run demonstrations
        await demo_web_scraping()
        await demo_performance_benchmarking()
        await demo_tool_benchmarking()
        await demo_advanced_agent()
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Some features may require additional setup or API keys")


if __name__ == "__main__":
    print("Agentic Framework - Advanced Features Demo")
    print("==========================================\n")
    
    asyncio.run(main())
