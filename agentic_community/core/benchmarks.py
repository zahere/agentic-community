"""
Performance benchmarking utilities for the Agentic Framework.

This module provides tools to measure and analyze the performance
of agents, tools, and API endpoints.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from statistics import mean, median, stdev

import psutil
from pydantic import BaseModel

from agentic_community.agents import SimpleAgent
from agentic_community.core.base import BaseTool


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    
    name: str
    duration: float  # seconds
    memory_used: float  # MB
    cpu_percent: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    

@dataclass
class BenchmarkSummary:
    """Summary statistics for multiple benchmark runs."""
    
    name: str
    runs: int
    avg_duration: float
    min_duration: float
    max_duration: float
    median_duration: float
    std_duration: float
    avg_memory: float
    avg_cpu: float
    success_rate: float
    errors: List[str]
    

class PerformanceBenchmark:
    """
    Performance benchmarking framework for agents and tools.
    
    Features:
    - Measure execution time, memory, and CPU usage
    - Run multiple iterations for statistical analysis
    - Compare different configurations
    - Generate performance reports
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize benchmark framework.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = output_dir or Path("benchmarks")
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
        
    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
        
    async def benchmark_function(
        self,
        func: Callable,
        name: str,
        iterations: int = 10,
        warmup: int = 2,
        **kwargs
    ) -> BenchmarkSummary:
        """
        Benchmark a function or coroutine.
        
        Args:
            func: Function to benchmark
            name: Name for the benchmark
            iterations: Number of times to run
            warmup: Number of warmup runs
            **kwargs: Arguments to pass to function
            
        Returns:
            Summary of benchmark results
        """
        results = []
        
        # Warmup runs
        for _ in range(warmup):
            if asyncio.iscoroutinefunction(func):
                await func(**kwargs)
            else:
                func(**kwargs)
                
        # Actual benchmark runs
        for i in range(iterations):
            start_memory = self._get_memory_usage()
            start_cpu = self._get_cpu_percent()
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    await func(**kwargs)
                else:
                    func(**kwargs)
                    
                duration = time.time() - start_time
                memory_used = self._get_memory_usage() - start_memory
                cpu_percent = self._get_cpu_percent() - start_cpu
                
                result = BenchmarkResult(
                    name=name,
                    duration=duration,
                    memory_used=memory_used,
                    cpu_percent=cpu_percent,
                    metadata={"iteration": i, "args": kwargs}
                )
                
            except Exception as e:
                duration = time.time() - start_time
                result = BenchmarkResult(
                    name=name,
                    duration=duration,
                    memory_used=0,
                    cpu_percent=0,
                    error=str(e),
                    metadata={"iteration": i, "args": kwargs}
                )
                
            results.append(result)
            self.results.append(result)
            
            # Small delay between runs
            await asyncio.sleep(0.1)
            
        return self._summarize_results(name, results)
        
    async def benchmark_agent(
        self,
        agent: SimpleAgent,
        tasks: List[str],
        name: Optional[str] = None,
        iterations: int = 5
    ) -> BenchmarkSummary:
        """
        Benchmark an agent with various tasks.
        
        Args:
            agent: Agent to benchmark
            tasks: List of tasks to run
            name: Name for the benchmark
            iterations: Number of iterations per task
            
        Returns:
            Summary of benchmark results
        """
        name = name or f"Agent_{agent.name}"
        all_results = []
        
        for task in tasks:
            print(f"Benchmarking task: {task[:50]}...")
            
            async def run_task():
                return await agent.run(task)
                
            task_results = await self.benchmark_function(
                run_task,
                f"{name}_task",
                iterations=iterations
            )
            
            all_results.extend(self.results[-iterations:])
            
        return self._summarize_results(name, all_results)
        
    async def benchmark_tool(
        self,
        tool: BaseTool,
        test_inputs: List[Dict[str, Any]],
        name: Optional[str] = None,
        iterations: int = 10
    ) -> BenchmarkSummary:
        """
        Benchmark a tool with various inputs.
        
        Args:
            tool: Tool to benchmark
            test_inputs: List of input dictionaries
            name: Name for the benchmark
            iterations: Number of iterations per input
            
        Returns:
            Summary of benchmark results
        """
        name = name or f"Tool_{tool.name}"
        all_results = []
        
        for input_data in test_inputs:
            print(f"Benchmarking input: {input_data}")
            
            async def run_tool():
                return await tool.execute(**input_data)
                
            input_results = await self.benchmark_function(
                run_tool,
                f"{name}_input",
                iterations=iterations
            )
            
            all_results.extend(self.results[-iterations:])
            
        return self._summarize_results(name, all_results)
        
    def _summarize_results(
        self,
        name: str,
        results: List[BenchmarkResult]
    ) -> BenchmarkSummary:
        """Calculate summary statistics from benchmark results."""
        successful_runs = [r for r in results if r.error is None]
        failed_runs = [r for r in results if r.error is not None]
        
        if not successful_runs:
            return BenchmarkSummary(
                name=name,
                runs=len(results),
                avg_duration=0,
                min_duration=0,
                max_duration=0,
                median_duration=0,
                std_duration=0,
                avg_memory=0,
                avg_cpu=0,
                success_rate=0,
                errors=[r.error for r in failed_runs]
            )
            
        durations = [r.duration for r in successful_runs]
        memories = [r.memory_used for r in successful_runs]
        cpus = [r.cpu_percent for r in successful_runs]
        
        return BenchmarkSummary(
            name=name,
            runs=len(results),
            avg_duration=mean(durations),
            min_duration=min(durations),
            max_duration=max(durations),
            median_duration=median(durations),
            std_duration=stdev(durations) if len(durations) > 1 else 0,
            avg_memory=mean(memories),
            avg_cpu=mean(cpus),
            success_rate=len(successful_runs) / len(results),
            errors=[r.error for r in failed_runs if r.error]
        )
        
    def compare_benchmarks(
        self,
        summaries: List[BenchmarkSummary]
    ) -> Dict[str, Any]:
        """
        Compare multiple benchmark summaries.
        
        Args:
            summaries: List of benchmark summaries to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            "fastest": min(summaries, key=lambda s: s.avg_duration),
            "slowest": max(summaries, key=lambda s: s.avg_duration),
            "most_memory": max(summaries, key=lambda s: s.avg_memory),
            "least_memory": min(summaries, key=lambda s: s.avg_memory),
            "rankings": sorted(summaries, key=lambda s: s.avg_duration)
        }
        
        # Calculate relative performance
        baseline = comparison["fastest"].avg_duration
        for summary in summaries:
            summary.metadata["relative_speed"] = summary.avg_duration / baseline
            
        return comparison
        
    def generate_report(
        self,
        summaries: List[BenchmarkSummary],
        filename: Optional[str] = None
    ) -> str:
        """
        Generate a performance report.
        
        Args:
            summaries: List of benchmark summaries
            filename: Optional filename to save report
            
        Returns:
            Report as string
        """
        report = "# Performance Benchmark Report\n\n"
        report += f"Generated: {datetime.now().isoformat()}\n\n"
        
        # Summary table
        report += "## Summary\n\n"
        report += "| Benchmark | Runs | Avg Duration (s) | Min | Max | Memory (MB) | Success Rate |\n"
        report += "|-----------|------|------------------|-----|-----|-------------|-------------|\n"
        
        for summary in summaries:
            report += f"| {summary.name} | {summary.runs} | "
            report += f"{summary.avg_duration:.3f} | "
            report += f"{summary.min_duration:.3f} | "
            report += f"{summary.max_duration:.3f} | "
            report += f"{summary.avg_memory:.1f} | "
            report += f"{summary.success_rate:.1%} |\n"
            
        # Comparison
        if len(summaries) > 1:
            report += "\n## Comparison\n\n"
            comparison = self.compare_benchmarks(summaries)
            
            report += f"- Fastest: {comparison['fastest'].name} "
            report += f"({comparison['fastest'].avg_duration:.3f}s)\n"
            report += f"- Slowest: {comparison['slowest'].name} "
            report += f"({comparison['slowest'].avg_duration:.3f}s)\n"
            report += f"- Most Memory: {comparison['most_memory'].name} "
            report += f"({comparison['most_memory'].avg_memory:.1f}MB)\n"
            
        # Errors
        report += "\n## Errors\n\n"
        for summary in summaries:
            if summary.errors:
                report += f"### {summary.name}\n"
                for error in set(summary.errors):
                    count = summary.errors.count(error)
                    report += f"- {error} (occurred {count} times)\n"
                report += "\n"
                
        # Save report
        if filename:
            report_path = self.output_dir / filename
            report_path.write_text(report)
            
        return report
        
    def save_results(self, filename: Optional[str] = None):
        """Save all benchmark results to JSON."""
        filename = filename or f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "name": r.name,
                    "duration": r.duration,
                    "memory_used": r.memory_used,
                    "cpu_percent": r.cpu_percent,
                    "timestamp": r.timestamp.isoformat(),
                    "metadata": r.metadata,
                    "error": r.error
                }
                for r in self.results
            ]
        }
        
        output_path = self.output_dir / filename
        output_path.write_text(json.dumps(data, indent=2))
        
        return output_path


# Convenience functions for quick benchmarking

async def quick_benchmark_agent(agent: SimpleAgent, task: str) -> BenchmarkSummary:
    """Quick benchmark of an agent with a single task."""
    benchmark = PerformanceBenchmark()
    return await benchmark.benchmark_agent(agent, [task], iterations=3)
    

async def quick_benchmark_tool(tool: BaseTool, **kwargs) -> BenchmarkSummary:
    """Quick benchmark of a tool with single input."""
    benchmark = PerformanceBenchmark()
    return await benchmark.benchmark_tool(tool, [kwargs], iterations=5)
