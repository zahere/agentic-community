"""
Memory management for agents.

This module provides conversation memory and context management for agents.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from pathlib import Path

from agentic_community.core.exceptions import ValidationError, StateError

logger = logging.getLogger(__name__)


class MemoryStore:
    """Base class for memory storage."""
    
    def save(self, agent_id: str, memory: Dict[str, Any]) -> None:
        """Save memory to storage."""
        raise NotImplementedError
    
    def load(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load memory from storage."""
        raise NotImplementedError
    
    def delete(self, agent_id: str) -> None:
        """Delete memory from storage."""
        raise NotImplementedError


class InMemoryStore(MemoryStore):
    """In-memory storage for agent memories."""
    
    def __init__(self):
        self.memories: Dict[str, Dict[str, Any]] = {}
    
    def save(self, agent_id: str, memory: Dict[str, Any]) -> None:
        """Save memory in-memory."""
        self.memories[agent_id] = memory
    
    def load(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load memory from in-memory storage."""
        return self.memories.get(agent_id)
    
    def delete(self, agent_id: str) -> None:
        """Delete memory from in-memory storage."""
        if agent_id in self.memories:
            del self.memories[agent_id]


class FileMemoryStore(MemoryStore):
    """File-based storage for agent memories."""
    
    def __init__(self, base_path: str = ".agent_memories"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def _get_path(self, agent_id: str) -> Path:
        """Get file path for agent memory."""
        return self.base_path / f"{agent_id}.json"
    
    def save(self, agent_id: str, memory: Dict[str, Any]) -> None:
        """Save memory to file."""
        path = self._get_path(agent_id)
        with open(path, 'w') as f:
            json.dump(memory, f, indent=2, default=str)
    
    def load(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load memory from file."""
        path = self._get_path(agent_id)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def delete(self, agent_id: str) -> None:
        """Delete memory file."""
        path = self._get_path(agent_id)
        if path.exists():
            path.unlink()


class ConversationMemory:
    """
    Manages conversation history and context for an agent.
    
    Supports different memory types:
    - Short-term: Recent conversation history
    - Long-term: Important facts and context
    - Working: Current task context
    """
    
    def __init__(
        self,
        agent_id: str,
        max_short_term: int = 10,
        max_long_term: int = 50,
        store: Optional[MemoryStore] = None
    ):
        """
        Initialize conversation memory.
        
        Args:
            agent_id: Unique identifier for the agent
            max_short_term: Maximum short-term memory items
            max_long_term: Maximum long-term memory items
            store: Memory storage backend
        """
        self.agent_id = agent_id
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        self.store = store or InMemoryStore()
        
        # Memory components
        self.short_term: deque = deque(maxlen=max_short_term)
        self.long_term: List[Dict[str, Any]] = []
        self.working_memory: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}
        
        # Load existing memory
        self._load()
    
    def add_interaction(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """
        Add an interaction to memory.
        
        Args:
            role: Role of the speaker (user, assistant, system)
            content: Content of the message
            metadata: Additional metadata
        """
        interaction = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add to short-term memory
        self.short_term.append(interaction)
        
        # Check if this should be promoted to long-term
        if self._is_important(interaction):
            self._add_to_long_term(interaction)
        
        # Save state
        self._save()
    
    def add_fact(self, fact: str, category: str = "general", confidence: float = 1.0) -> None:
        """
        Add a fact to long-term memory.
        
        Args:
            fact: The fact to remember
            category: Category of the fact
            confidence: Confidence level (0-1)
        """
        fact_entry = {
            "type": "fact",
            "content": fact,
            "category": category,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._add_to_long_term(fact_entry)
        self._save()
    
    def update_context(self, key: str, value: Any) -> None:
        """
        Update conversation context.
        
        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value
        self._save()
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get context value.
        
        Args:
            key: Context key
            default: Default value if key not found
            
        Returns:
            Context value
        """
        return self.context.get(key, default)
    
    def set_working_memory(self, key: str, value: Any) -> None:
        """
        Set working memory item.
        
        Args:
            key: Memory key
            value: Memory value
        """
        self.working_memory[key] = value
        self._save()
    
    def get_working_memory(self, key: str, default: Any = None) -> Any:
        """
        Get working memory item.
        
        Args:
            key: Memory key
            default: Default value
            
        Returns:
            Memory value
        """
        return self.working_memory.get(key, default)
    
    def clear_working_memory(self) -> None:
        """Clear working memory."""
        self.working_memory.clear()
        self._save()
    
    def get_recent_history(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent conversation history.
        
        Args:
            n: Number of recent interactions
            
        Returns:
            List of recent interactions
        """
        return list(self.short_term)[-n:]
    
    def search_memory(self, query: str, memory_type: str = "all") -> List[Dict[str, Any]]:
        """
        Search memory for relevant information.
        
        Args:
            query: Search query
            memory_type: Type of memory to search (short, long, all)
            
        Returns:
            List of matching memory items
        """
        results = []
        query_lower = query.lower()
        
        # Search short-term
        if memory_type in ["short", "all"]:
            for item in self.short_term:
                if query_lower in item.get("content", "").lower():
                    results.append({"source": "short_term", **item})
        
        # Search long-term
        if memory_type in ["long", "all"]:
            for item in self.long_term:
                content = item.get("content", "")
                if isinstance(content, str) and query_lower in content.lower():
                    results.append({"source": "long_term", **item})
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get memory summary.
        
        Returns:
            Summary of memory contents
        """
        return {
            "agent_id": self.agent_id,
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "working_memory_keys": list(self.working_memory.keys()),
            "context_keys": list(self.context.keys()),
            "oldest_memory": self._get_oldest_timestamp(),
            "newest_memory": self._get_newest_timestamp()
        }
    
    def _is_important(self, interaction: Dict[str, Any]) -> bool:
        """
        Determine if an interaction is important enough for long-term memory.
        
        Args:
            interaction: The interaction to evaluate
            
        Returns:
            True if important
        """
        # Simple heuristics for importance
        content = interaction.get("content", "").lower()
        
        # Keywords that indicate importance
        important_keywords = [
            "remember", "important", "don't forget", "note that",
            "key point", "summary", "conclusion", "decision"
        ]
        
        for keyword in important_keywords:
            if keyword in content:
                return True
        
        # Long messages might be important
        if len(content) > 200:
            return True
        
        # User corrections are important
        if "correction" in interaction.get("metadata", {}):
            return True
        
        return False
    
    def _add_to_long_term(self, item: Dict[str, Any]) -> None:
        """Add item to long-term memory with size management."""
        self.long_term.append(item)
        
        # Manage size
        if len(self.long_term) > self.max_long_term:
            # Remove oldest items
            self.long_term = self.long_term[-self.max_long_term:]
    
    def _get_oldest_timestamp(self) -> Optional[str]:
        """Get timestamp of oldest memory item."""
        timestamps = []
        
        for item in self.short_term:
            if "timestamp" in item:
                timestamps.append(item["timestamp"])
        
        for item in self.long_term:
            if "timestamp" in item:
                timestamps.append(item["timestamp"])
        
        return min(timestamps) if timestamps else None
    
    def _get_newest_timestamp(self) -> Optional[str]:
        """Get timestamp of newest memory item."""
        timestamps = []
        
        for item in self.short_term:
            if "timestamp" in item:
                timestamps.append(item["timestamp"])
        
        for item in self.long_term:
            if "timestamp" in item:
                timestamps.append(item["timestamp"])
        
        return max(timestamps) if timestamps else None
    
    def _save(self) -> None:
        """Save memory to storage."""
        memory_data = {
            "agent_id": self.agent_id,
            "short_term": list(self.short_term),
            "long_term": self.long_term,
            "working_memory": self.working_memory,
            "context": self.context,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            self.store.save(self.agent_id, memory_data)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def _load(self) -> None:
        """Load memory from storage."""
        try:
            memory_data = self.store.load(self.agent_id)
            if memory_data:
                # Restore short-term memory
                short_term_data = memory_data.get("short_term", [])
                self.short_term = deque(short_term_data, maxlen=self.max_short_term)
                
                # Restore other components
                self.long_term = memory_data.get("long_term", [])
                self.working_memory = memory_data.get("working_memory", {})
                self.context = memory_data.get("context", {})
                
                logger.info(f"Loaded memory for agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")


class MemoryManager:
    """
    Manages memory for multiple agents.
    """
    
    def __init__(self, store: Optional[MemoryStore] = None):
        """
        Initialize memory manager.
        
        Args:
            store: Memory storage backend
        """
        self.store = store or InMemoryStore()
        self.memories: Dict[str, ConversationMemory] = {}
    
    def get_memory(self, agent_id: str) -> ConversationMemory:
        """
        Get or create memory for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            ConversationMemory instance
        """
        if agent_id not in self.memories:
            self.memories[agent_id] = ConversationMemory(
                agent_id=agent_id,
                store=self.store
            )
        
        return self.memories[agent_id]
    
    def delete_memory(self, agent_id: str) -> None:
        """
        Delete memory for an agent.
        
        Args:
            agent_id: Agent identifier
        """
        if agent_id in self.memories:
            del self.memories[agent_id]
        
        self.store.delete(agent_id)


# Global memory manager
memory_manager = MemoryManager()
