"""Tests for REST API functionality."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from agentic_community.api import app


client = TestClient(app)


class TestAPI:
    """Test REST API endpoints."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    @patch('os.environ.get')
    def test_create_agent(self, mock_env):
        """Test agent creation endpoint."""
        mock_env.return_value = "mock-api-key"
        
        response = client.post(
            "/agents/create",
            json={"name": "TestAgent", "tools": ["calculator"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "TestAgent"
        assert "agent_id" in data
    
    @patch('os.environ.get')
    def test_run_task(self, mock_env):
        """Test running a task via API."""
        mock_env.return_value = "mock-api-key"
        
        # First create an agent
        create_response = client.post(
            "/agents/create",
            json={"name": "TestAgent", "tools": ["calculator"]}
        )
        agent_id = create_response.json()["agent_id"]
        
        # Mock the agent execution
        with patch('agentic_community.api.agents') as mock_agents:
            mock_agent = MagicMock()
            mock_agent.run.return_value = "The answer is 5"
            mock_agents[agent_id] = mock_agent
            
            # Run a task
            response = client.post(
                f"/agents/{agent_id}/run",
                json={"task": "What is 2 + 3?"}
            )
            assert response.status_code == 200
            data = response.json()
            assert "result" in data
    
    def test_invalid_agent_id(self):
        """Test handling of invalid agent ID."""
        response = client.post(
            "/agents/invalid-id/run",
            json={"task": "Test task"}
        )
        assert response.status_code == 404
    
    def test_missing_task(self):
        """Test handling of missing task in request."""
        response = client.post(
            "/agents/test-id/run",
            json={}
        )
        assert response.status_code == 422  # Validation error
    
    @patch('os.environ.get')
    def test_get_agent_state(self, mock_env):
        """Test getting agent state."""
        mock_env.return_value = "mock-api-key"
        
        # Create an agent
        create_response = client.post(
            "/agents/create",
            json={"name": "TestAgent", "tools": []}
        )
        agent_id = create_response.json()["agent_id"]
        
        # Mock the agent
        with patch('agentic_community.api.agents') as mock_agents:
            mock_agent = MagicMock()
            mock_agent.get_state.return_value = {
                "name": "TestAgent",
                "tools": [],
                "history": []
            }
            mock_agents[agent_id] = mock_agent
            
            # Get state
            response = client.get(f"/agents/{agent_id}/state")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "TestAgent"
    
    @patch('os.environ.get')
    def test_delete_agent(self, mock_env):
        """Test deleting an agent."""
        mock_env.return_value = "mock-api-key"
        
        # Create an agent
        create_response = client.post(
            "/agents/create",
            json={"name": "TestAgent", "tools": []}
        )
        agent_id = create_response.json()["agent_id"]
        
        # Delete it
        with patch('agentic_community.api.agents') as mock_agents:
            mock_agents[agent_id] = MagicMock()
            
            response = client.delete(f"/agents/{agent_id}")
            assert response.status_code == 200
            data = response.json()
            assert "deleted" in data["message"].lower()
