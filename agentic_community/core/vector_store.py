"""
Qdrant vector database integration for semantic search and RAG.

Provides high-performance vector storage and retrieval for agent memory,
knowledge bases, and retrieval-augmented generation (RAG) workflows.
"""

import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
import logging
import json

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, 
        Filter, FieldCondition, MatchValue,
        SearchRequest, UpdateStatus
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Qdrant client not installed. Install with: pip install qdrant-client")

from ..base import BaseTool
from ...agents.base import BaseAgent


@dataclass
class Document:
    """Document for vector storage."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow()


@dataclass
class SearchResult:
    """Result from vector search."""
    document: Document
    score: float
    rank: int


class VectorStore:
    """Abstract interface for vector stores."""
    
    def add_documents(self, documents: List[Document], **kwargs):
        """Add documents to the vector store."""
        raise NotImplementedError
        
    def search(self, query: Union[str, List[float]], k: int = 10, **kwargs) -> List[SearchResult]:
        """Search for similar documents."""
        raise NotImplementedError
        
    def delete(self, ids: List[str]):
        """Delete documents by ID."""
        raise NotImplementedError
        
    def update(self, documents: List[Document]):
        """Update existing documents."""
        raise NotImplementedError


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of vector store."""
    
    def __init__(
        self,
        collection_name: str = "agentic_vectors",
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        vector_size: int = 1536,  # OpenAI embedding size
        distance: str = "cosine"
    ):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not installed")
            
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = Distance.COSINE if distance == "cosine" else Distance.EUCLID
        
        # Initialize client
        self.client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key
        )
        
        # Create collection if it doesn't exist
        self._ensure_collection()
        
    def _ensure_collection(self):
        """Ensure the collection exists."""
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                )
            )
            
    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """Add documents to Qdrant."""
        points = []
        
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} has no embedding")
                
            point = PointStruct(
                id=doc.id,
                vector=doc.embedding,
                payload={
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "timestamp": doc.timestamp.isoformat() if doc.timestamp else None
                }
            )
            points.append(point)
            
            # Batch upload
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                points = []
                
        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
    def search(
        self, 
        query: Union[str, List[float]], 
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Search for similar documents in Qdrant."""
        # If query is a string, it should be embedded first
        if isinstance(query, str):
            raise ValueError("String queries must be embedded first")
            
        # Build filter if provided
        qdrant_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)
                
        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query,
            limit=k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold
        )
        
        # Convert to SearchResult objects
        search_results = []
        for i, result in enumerate(results):
            doc = Document(
                id=str(result.id),
                content=result.payload.get("content", ""),
                metadata=result.payload.get("metadata", {}),
                embedding=None,  # Don't return embeddings in search
                timestamp=datetime.fromisoformat(result.payload["timestamp"])
                if result.payload.get("timestamp") else None
            )
            
            search_results.append(SearchResult(
                document=doc,
                score=result.score,
                rank=i + 1
            ))
            
        return search_results
        
    def delete(self, ids: List[str]):
        """Delete documents from Qdrant."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )
        
    def update(self, documents: List[Document]):
        """Update documents in Qdrant."""
        # Qdrant upsert handles updates
        self.add_documents(documents)
        
    def create_index(self, field: str):
        """Create an index on a payload field for faster filtering."""
        # Qdrant automatically indexes payload fields
        pass
        
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vector_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
            "points_count": info.points_count
        }


class SemanticMemory:
    """Semantic memory system using vector database."""
    
    def __init__(self, vector_store: VectorStore, embedding_function: callable):
        self.vector_store = vector_store
        self.embed = embedding_function
        
    def remember(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Store a memory in the vector database."""
        embedding = self.embed(content)
        doc = Document(
            id=str(uuid.uuid4()),
            content=content,
            metadata=metadata or {},
            embedding=embedding
        )
        self.vector_store.add_documents([doc])
        return doc.id
        
    def recall(
        self, 
        query: str, 
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Recall similar memories."""
        query_embedding = self.embed(query)
        results = self.vector_store.search(
            query=query_embedding,
            k=k,
            filter_dict=filter_dict
        )
        return [r.document for r in results]
        
    def forget(self, memory_ids: List[str]):
        """Remove memories from the database."""
        self.vector_store.delete(memory_ids)


class RAGTool(BaseTool):
    """Retrieval-Augmented Generation tool using vector database."""
    
    name = "rag_search"
    description = "Search knowledge base using semantic similarity"
    
    def __init__(self, vector_store: VectorStore, embedding_function: callable):
        super().__init__()
        self.vector_store = vector_store
        self.embed = embedding_function
        
    def _run(self, query: str, k: int = 5) -> str:
        """Search the knowledge base and return relevant context."""
        query_embedding = self.embed(query)
        results = self.vector_store.search(query_embedding, k=k)
        
        # Format results for context
        context_parts = []
        for result in results:
            context_parts.append(f"[Score: {result.score:.3f}] {result.document.content}")
            
        return "\n\n".join(context_parts)


class KnowledgeBase:
    """Knowledge base management using vector database."""
    
    def __init__(self, vector_store: VectorStore, embedding_function: callable):
        self.vector_store = vector_store
        self.embed = embedding_function
        
    def add_knowledge(
        self, 
        documents: List[str],
        source: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """Add documents to the knowledge base with chunking."""
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            # Simple chunking - in production use better strategies
            chunks = self._chunk_text(doc, chunk_size, chunk_overlap)
            
            for chunk_idx, chunk in enumerate(chunks):
                embedding = self.embed(chunk)
                doc_obj = Document(
                    id=f"{source}_{doc_idx}_{chunk_idx}" if source else str(uuid.uuid4()),
                    content=chunk,
                    metadata={
                        "source": source,
                        "doc_index": doc_idx,
                        "chunk_index": chunk_idx
                    },
                    embedding=embedding
                )
                all_chunks.append(doc_obj)
                
        self.vector_store.add_documents(all_chunks)
        return len(all_chunks)
        
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Simple text chunking."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
        return chunks
        
    def query(
        self, 
        query: str, 
        k: int = 5,
        source_filter: Optional[str] = None
    ) -> List[Document]:
        """Query the knowledge base."""
        filter_dict = {"source": source_filter} if source_filter else None
        query_embedding = self.embed(query)
        results = self.vector_store.search(
            query=query_embedding,
            k=k,
            filter_dict=filter_dict
        )
        return [r.document for r in results]


# Example embedding function (placeholder)
def mock_embedding_function(text: str) -> List[float]:
    """Mock embedding function for testing."""
    # In production, use OpenAI, Cohere, or other embedding models
    import hashlib
    
    # Create deterministic "embedding" from text
    hash_obj = hashlib.md5(text.encode())
    hash_bytes = hash_obj.digest()
    
    # Convert to float vector
    embedding = []
    for i in range(0, len(hash_bytes), 2):
        value = int.from_bytes(hash_bytes[i:i+2], 'big') / 65535.0
        embedding.append(value)
        
    # Pad to standard size (1536 for OpenAI)
    while len(embedding) < 1536:
        embedding.append(0.0)
        
    return embedding[:1536]
