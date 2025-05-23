"""Integration tests for agent workflows."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from agentic_community import SimpleAgent
from agentic_community.tools import SearchTool, CalculatorTool, TextTool
from agentic_community.core.exceptions import AgentExecutionError, InvalidTaskError


class TestAgentIntegration:
    """Test complete agent workflows with multiple tools."""
    
    @pytest.fixture
    def agent_with_tools(self):
        """Create an agent with all available tools."""
        agent = SimpleAgent("IntegrationTestAgent")
        agent.add_tool(SearchTool())
        agent.add_tool(CalculatorTool())
        agent.add_tool(TextTool())
        return agent
    
    def test_simple_calculation_workflow(self, agent_with_tools):
        """Test a simple calculation workflow."""
        task = "Calculate the total cost: 3 items at $25.99 each, plus 8% tax"
        
        # Mock LLM to return calculation steps
        with patch.object(agent_with_tools, 'llm') as mock_llm:
            mock_response = MagicMock()
            mock_response.content = (
                "Step 1: Calculate subtotal (3 * 25.99)\n"
                "Step 2: Calculate tax (subtotal * 0.08)\n"
                "Step 3: Calculate total (subtotal + tax)"
            )
            mock_llm.invoke.return_value = mock_response
            
            result = agent_with_tools.run(task)
            
            assert result is not None
            assert "Step" in result
            assert len(agent_with_tools.state_manager.get_history(agent_with_tools.config.name)) > 0
    
    def test_research_workflow(self, agent_with_tools):
        """Test a research workflow with search and text processing."""
        task = "Search for information about Python programming and summarize it"
        
        with patch.object(agent_with_tools, 'llm') as mock_llm:
            mock_response = MagicMock()
            mock_response.content = (
                "Step 1: Search for Python programming information\n"
                "Step 2: Extract key points from search results\n"
                "Step 3: Summarize the findings"
            )
            mock_llm.invoke.return_value = mock_response
            
            result = agent_with_tools.run(task)
            
            assert result is not None
            assert "Step" in result
    
    def test_multi_tool_workflow(self, agent_with_tools):
        """Test workflow using multiple tools in sequence."""
        task = "Search for the current price of gold, calculate how much 5 ounces would cost, and format the result nicely"
        
        with patch.object(agent_with_tools, 'llm') as mock_llm:
            # First response for planning
            mock_response1 = MagicMock()
            mock_response1.content = (
                "Step 1: Search for current gold price\n"
                "Step 2: Calculate cost of 5 ounces\n"
                "Step 3: Format the result with text processing"
            )
            mock_llm.invoke.return_value = mock_response1
            
            result = agent_with_tools.run(task)
            
            assert result is not None
            assert "Completed" in result
    
    def test_error_recovery_workflow(self, agent_with_tools):
        """Test that agent handles errors gracefully."""
        task = "Do something that will fail"
        
        # Make tool execution fail
        with patch.object(agent_with_tools.tools[0], 'invoke', side_effect=Exception("Tool failed")):
            result = agent_with_tools.run(task)
            
            # Should complete but note the error
            assert result is not None
            assert "error" in result.lower() or "failed" in result.lower()
    
    def test_invalid_task_handling(self, agent_with_tools):
        """Test handling of invalid tasks."""
        # Empty task
        with pytest.raises(InvalidTaskError):
            agent_with_tools.run("")
        
        # Too short task
        with pytest.raises(InvalidTaskError):
            agent_with_tools.run("Hi")
    
    def test_state_persistence_workflow(self, agent_with_tools):
        """Test that agent state persists across multiple tasks."""
        # First task
        task1 = "Calculate 10 + 20"
        result1 = agent_with_tools.run(task1)
        
        # Get state
        state1 = agent_with_tools.get_state()
        assert state1["name"] == "IntegrationTestAgent"
        
        # Second task
        task2 = "Calculate 30 + 40"
        result2 = agent_with_tools.run(task2)
        
        # Check history has both tasks
        history = agent_with_tools.state_manager.get_history(agent_with_tools.config.name)
        assert len(history) >= 2
        
        # Clear history
        agent_with_tools.clear_history()
        history_after = agent_with_tools.state_manager.get_history(agent_with_tools.config.name)
        assert len(history_after) == 0
    
    @pytest.mark.asyncio
    async def test_async_search_integration(self):
        """Test async search functionality."""
        search_tool = SearchTool()
        
        # Mock the aiohttp response
        with patch('agentic_community.tools.search_tool.aiohttp.ClientSession') as mock_session:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.text = asyncio.coroutine(lambda: "<html><body>Test results</body></html>")()
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            results = await search_tool.search_async("test query")
            assert isinstance(results, list)
        
        await search_tool.close()
    
    def test_tool_limit_enforcement(self):
        """Test that community edition enforces tool limits."""
        agent = SimpleAgent("LimitTestAgent")
        
        # Add 3 tools (should work)
        agent.add_tool(SearchTool())
        agent.add_tool(CalculatorTool())
        agent.add_tool(TextTool())
        
        # Try to add 4th tool (should fail)
        from agentic_community.core.exceptions import ToolLimitExceeded
        with pytest.raises(ToolLimitExceeded):
            agent.add_tool(SearchTool())  # Duplicate, but should fail due to limit
    
    def test_concurrent_agent_execution(self):
        """Test multiple agents can work concurrently."""
        agent1 = SimpleAgent("Agent1")
        agent2 = SimpleAgent("Agent2")
        
        agent1.add_tool(CalculatorTool())
        agent2.add_tool(TextTool())
        
        # Run tasks
        result1 = agent1.run("Calculate 5 + 5")
        result2 = agent2.run("Count words in: Hello world")
        
        # Both should complete successfully
        assert result1 is not None
        assert result2 is not None
        
        # States should be independent
        state1 = agent1.get_state()
        state2 = agent2.get_state()
        assert state1["name"] == "Agent1"
        assert state2["name"] == "Agent2"


class TestEndToEndScenarios:
    """Test real-world end-to-end scenarios."""
    
    def test_trip_planning_scenario(self):
        """Test a trip planning scenario."""
        agent = SimpleAgent("TripPlanner")
        agent.add_tool(SearchTool())
        agent.add_tool(CalculatorTool())
        agent.add_tool(TextTool())
        
        task = """
        Help me plan a trip to Paris:
        1. Search for popular attractions
        2. Calculate a daily budget ($200/day for 5 days)
        3. Create a summary
        """
        
        with patch.object(agent, 'llm') as mock_llm:
            mock_response = MagicMock()
            mock_response.content = (
                "Step 1: Search for Paris attractions\n"
                "Step 2: Calculate total budget (200 * 5)\n"
                "Step 3: Create trip summary with attractions and budget"
            )
            mock_llm.invoke.return_value = mock_response
            
            result = agent.run(task)
            
            assert result is not None
            assert "Step" in result
            assert "Completed" in result
    
    def test_data_analysis_scenario(self):
        """Test a data analysis scenario."""
        agent = SimpleAgent("DataAnalyst")
        agent.add_tool(CalculatorTool())
        agent.add_tool(TextTool())
        
        task = """
        Analyze these sales figures:
        - Q1: $45,000
        - Q2: $52,000
        - Q3: $48,000
        - Q4: $61,000
        Calculate the total and average, then summarize the trend.
        """
        
        result = agent.run(task)
        
        assert result is not None
        # Should have performed calculations
        assert "Step" in result
