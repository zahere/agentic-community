"""
Agents module for Agentic Community Edition
"""

# Try to import the real SimpleAgent, fall back to mock if needed
try:
    from .simple_agent import SimpleAgent
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import SimpleAgent with LangChain: {e}. Using mock version.")
    from .simple_agent_mock import SimpleAgent

__all__ = ['SimpleAgent']
