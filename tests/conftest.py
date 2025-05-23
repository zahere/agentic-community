"""Pytest configuration and fixtures."""

import pytest
import os
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_openai_key():
    """Automatically mock OpenAI API key for all tests."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


@pytest.fixture
def mock_llm_response():
    """Fixture to mock LLM responses."""
    def _mock_response(response="This is a mock response"):
        mock = patch('langchain_openai.ChatOpenAI')
        mock_instance = mock.start()
        mock_instance.return_value.invoke.return_value.content = response
        return mock
    
    yield _mock_response
    # Cleanup is handled by patch context manager


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path
