"""
Plugin system for extending Agentic Framework with custom tools and agents.

Provides a flexible plugin architecture that allows users to create and
register custom tools without modifying the core framework.
"""

import os
import sys
import json
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Callable
from dataclasses import dataclass, field
import logging

from ..core.base import BaseTool
from ..agents.base import BaseAgent
from ..core.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    author_email: Optional[str] = None
    url: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    min_framework_version: Optional[str] = None
    max_framework_version: Optional[str] = None


@dataclass
class PluginInfo:
    """Information about a loaded plugin."""
    metadata: PluginMetadata
    module_path: str
    tools: Dict[str, Type[BaseTool]] = field(default_factory=dict)
    agents: Dict[str, Type[BaseAgent]] = field(default_factory=dict)
    hooks: Dict[str, List[Callable]] = field(default_factory=dict)
    loaded_at: Optional[str] = None


class PluginRegistry:
    """Registry for managing plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
        self._tool_registry: Dict[str, Type[BaseTool]] = {}
        self._agent_registry: Dict[str, Type[BaseAgent]] = {}
        self._hook_registry: Dict[str, List[Callable]] = {}
        
    def register_plugin(self, plugin_info: PluginInfo):
        """Register a plugin."""
        if plugin_info.metadata.name in self._plugins:
            raise ValueError(f"Plugin '{plugin_info.metadata.name}' already registered")
            
        self._plugins[plugin_info.metadata.name] = plugin_info
        
        # Register tools
        for tool_name, tool_class in plugin_info.tools.items():
            self.register_tool(tool_name, tool_class, plugin_info.metadata.name)
            
        # Register agents
        for agent_name, agent_class in plugin_info.agents.items():
            self.register_agent(agent_name, agent_class, plugin_info.metadata.name)
            
        # Register hooks
        for hook_name, callbacks in plugin_info.hooks.items():
            for callback in callbacks:
                self.register_hook(hook_name, callback)
                
        logger.info(f"Registered plugin: {plugin_info.metadata.name} v{plugin_info.metadata.version}")
        
    def register_tool(self, name: str, tool_class: Type[BaseTool], plugin_name: str):
        """Register a tool from a plugin."""
        if name in self._tool_registry:
            logger.warning(f"Tool '{name}' already registered, overwriting")
            
        self._tool_registry[name] = tool_class
        logger.debug(f"Registered tool '{name}' from plugin '{plugin_name}'")
        
    def register_agent(self, name: str, agent_class: Type[BaseAgent], plugin_name: str):
        """Register an agent from a plugin."""
        if name in self._agent_registry:
            logger.warning(f"Agent '{name}' already registered, overwriting")
            
        self._agent_registry[name] = agent_class
        logger.debug(f"Registered agent '{name}' from plugin '{plugin_name}'")
        
    def register_hook(self, hook_name: str, callback: Callable):
        """Register a hook callback."""
        if hook_name not in self._hook_registry:
            self._hook_registry[hook_name] = []
            
        self._hook_registry[hook_name].append(callback)
        logger.debug(f"Registered hook '{hook_name}'")
        
    def get_tool(self, name: str) -> Optional[Type[BaseTool]]:
        """Get a tool class by name."""
        return self._tool_registry.get(name)
        
    def get_agent(self, name: str) -> Optional[Type[BaseAgent]]:
        """Get an agent class by name."""
        return self._agent_registry.get(name)
        
    def get_plugin(self, name: str) -> Optional[PluginInfo]:
        """Get plugin info by name."""
        return self._plugins.get(name)
        
    def list_plugins(self) -> List[PluginInfo]:
        """List all registered plugins."""
        return list(self._plugins.values())
        
    def list_tools(self) -> Dict[str, Type[BaseTool]]:
        """List all registered tools."""
        return self._tool_registry.copy()
        
    def list_agents(self) -> Dict[str, Type[BaseAgent]]:
        """List all registered agents."""
        return self._agent_registry.copy()
        
    def trigger_hook(self, hook_name: str, *args, **kwargs):
        """Trigger all callbacks for a hook."""
        if hook_name not in self._hook_registry:
            return
            
        for callback in self._hook_registry[hook_name]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in hook '{hook_name}': {e}")


# Global plugin registry
_plugin_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _plugin_registry


class PluginLoader:
    """Loads plugins from various sources."""
    
    @staticmethod
    def load_from_module(module_path: str) -> PluginInfo:
        """Load a plugin from a Python module."""
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location("plugin", module_path)
            if not spec or not spec.loader:
                raise ImportError(f"Cannot load module from {module_path}")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get plugin metadata
            if not hasattr(module, 'PLUGIN_METADATA'):
                raise ValueError("Plugin must define PLUGIN_METADATA")
                
            metadata_dict = module.PLUGIN_METADATA
            metadata = PluginMetadata(**metadata_dict)
            
            # Create plugin info
            plugin_info = PluginInfo(
                metadata=metadata,
                module_path=module_path
            )
            
            # Discover tools
            if hasattr(module, 'TOOLS'):
                for tool_name, tool_class in module.TOOLS.items():
                    if issubclass(tool_class, BaseTool):
                        plugin_info.tools[tool_name] = tool_class
                        
            # Discover agents
            if hasattr(module, 'AGENTS'):
                for agent_name, agent_class in module.AGENTS.items():
                    if issubclass(agent_class, BaseAgent):
                        plugin_info.agents[agent_name] = agent_class
                        
            # Discover hooks
            if hasattr(module, 'HOOKS'):
                plugin_info.hooks = module.HOOKS
                
            return plugin_info
            
        except Exception as e:
            logger.error(f"Failed to load plugin from {module_path}: {e}")
            raise
            
    @staticmethod
    def load_from_directory(directory: str) -> List[PluginInfo]:
        """Load all plugins from a directory."""
        plugins = []
        plugin_dir = Path(directory)
        
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return plugins
            
        # Look for plugin.py files
        for plugin_file in plugin_dir.glob("*/plugin.py"):
            try:
                plugin_info = PluginLoader.load_from_module(str(plugin_file))
                plugins.append(plugin_info)
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_file}: {e}")
                
        # Look for single-file plugins
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            try:
                plugin_info = PluginLoader.load_from_module(str(plugin_file))
                plugins.append(plugin_info)
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_file}: {e}")
                
        return plugins
        
    @staticmethod
    def load_from_package(package_name: str) -> PluginInfo:
        """Load a plugin from an installed package."""
        try:
            module = importlib.import_module(package_name)
            
            # Get plugin metadata
            if not hasattr(module, 'PLUGIN_METADATA'):
                raise ValueError("Plugin must define PLUGIN_METADATA")
                
            metadata_dict = module.PLUGIN_METADATA
            metadata = PluginMetadata(**metadata_dict)
            
            # Create plugin info
            plugin_info = PluginInfo(
                metadata=metadata,
                module_path=package_name
            )
            
            # Discover tools, agents, and hooks
            if hasattr(module, 'TOOLS'):
                plugin_info.tools = module.TOOLS
            if hasattr(module, 'AGENTS'):
                plugin_info.agents = module.AGENTS
            if hasattr(module, 'HOOKS'):
                plugin_info.hooks = module.HOOKS
                
            return plugin_info
            
        except Exception as e:
            logger.error(f"Failed to load plugin package {package_name}: {e}")
            raise


def load_plugin(source: str, source_type: str = "auto") -> PluginInfo:
    """
    Load a plugin from a source.
    
    Args:
        source: Path to module, directory, or package name
        source_type: Type of source ("module", "directory", "package", or "auto")
        
    Returns:
        PluginInfo object
    """
    if source_type == "auto":
        # Auto-detect source type
        if os.path.isfile(source) and source.endswith(".py"):
            source_type = "module"
        elif os.path.isdir(source):
            source_type = "directory"
        else:
            source_type = "package"
            
    if source_type == "module":
        plugin_info = PluginLoader.load_from_module(source)
    elif source_type == "directory":
        plugins = PluginLoader.load_from_directory(source)
        if not plugins:
            raise ValueError(f"No plugins found in directory: {source}")
        if len(plugins) > 1:
            logger.warning(f"Multiple plugins found in directory, loading first one")
        plugin_info = plugins[0]
    elif source_type == "package":
        plugin_info = PluginLoader.load_from_package(source)
    else:
        raise ValueError(f"Unknown source type: {source_type}")
        
    # Register the plugin
    registry = get_plugin_registry()
    registry.register_plugin(plugin_info)
    
    return plugin_info


def create_plugin_template(output_dir: str, plugin_name: str):
    """Create a plugin template for users to start with."""
    template = '''"""
{plugin_name} Plugin for Agentic Framework

This is a template plugin that demonstrates how to create custom tools
and agents for the Agentic Framework.
"""

from agentic_community.core.base import BaseTool
from agentic_community.agents.base import BaseAgent


# Plugin metadata (required)
PLUGIN_METADATA = {{
    "name": "{plugin_name}",
    "version": "1.0.0",
    "description": "A custom plugin for Agentic Framework",
    "author": "Your Name",
    "author_email": "your.email@example.com",
    "tags": ["custom", "example"],
    "dependencies": [],
}}


# Custom tool example
class CustomTool(BaseTool):
    """Example custom tool."""
    
    name = "custom_tool"
    description = "A custom tool that does something useful"
    
    def _run(self, input_data: str) -> str:
        """Execute the tool."""
        # Implement your tool logic here
        return f"Processed: {{input_data}}"


# Custom agent example (optional)
class CustomAgent(BaseAgent):
    """Example custom agent."""
    
    def __init__(self, name: str = "CustomAgent"):
        super().__init__(name)
        # Add any custom initialization
        
    def process(self, task: str) -> str:
        """Process a task."""
        # Implement your agent logic here
        return f"Custom agent processed: {{task}}"


# Export tools and agents (required)
TOOLS = {{
    "CustomTool": CustomTool,
}}

AGENTS = {{
    "CustomAgent": CustomAgent,
}}

# Hooks (optional)
HOOKS = {{
    "before_tool_run": [
        lambda tool, input_data: print(f"Running tool {{tool.name}} with {{input_data}}")
    ],
}}
'''
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write plugin file
    plugin_file = output_path / f"{plugin_name.lower()}_plugin.py"
    plugin_file.write_text(template.format(plugin_name=plugin_name))
    
    # Create README
    readme_content = f"""# {plugin_name} Plugin

This is a custom plugin for the Agentic Framework.

## Installation

1. Place this plugin in your plugins directory
2. Load it using the plugin loader:

```python
from agentic_community.plugins import load_plugin

plugin = load_plugin("path/to/{plugin_name.lower()}_plugin.py")
```

## Usage

```python
from agentic_community.plugins import get_plugin_registry

registry = get_plugin_registry()
CustomTool = registry.get_tool("CustomTool")

tool = CustomTool()
result = tool.run("test input")
```
"""
    
    readme_file = output_path / "README.md"
    readme_file.write_text(readme_content)
    
    logger.info(f"Created plugin template at: {output_path}")
    
    return str(plugin_file)
