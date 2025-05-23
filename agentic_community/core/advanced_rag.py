"""
Advanced RAG (Retrieval-Augmented Generation) Techniques

This module implements state-of-the-art RAG techniques including:
- HyDE (Hypothetical Document Embeddings)
- Multi-Query Retrieval
- Contextual Compression
- Fusion Retrieval
- Recursive Retrieval
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict

from agentic_community.core.vector_store import VectorStore
from agentic_community.core.llm_providers import create_llm_client, LLMResponse
from agentic_community.core.cache import cache_result
from agentic_community.core.exceptions import RAGError
from agentic_community.core.validation import validate_input


class RAGStrategy(Enum):
    """Available RAG strategies"""
    STANDARD = "standard"
    HYDE = "hyde"
    MULTI_QUERY = "multi_query"
    CONTEXTUAL_COMPRESSION = "contextual_compression"
    FUSION = "fusion"
    RECURSIVE = "recursive"
    HIERARCHICAL = "hierarchical"


@dataclass
class Document:
    """Represents a document in the RAG system"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score
        }


@dataclass
class RAGConfig:
    """Configuration for RAG operations"""
    strategy: RAGStrategy = RAGStrategy.STANDARD
    vector_store: Optional[VectorStore] = None
    llm_provider: str = "openai"
    llm_model: Optional[str] = None
    embedding_model: str = "text-embedding-ada-002"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    rerank_top_k: int = 3
    temperature: float = 0.7
    use_cache: bool = True
    cache_ttl: int = 3600


class AdvancedRAG:
    """
    Advanced RAG implementation with multiple retrieval strategies
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = config.vector_store or VectorStore()
        self.llm_client = create_llm_client(
            config.llm_provider,
            model=config.llm_model
        )
        
        # Strategy implementations
        self.strategies = {
            RAGStrategy.STANDARD: self._standard_rag,
            RAGStrategy.HYDE: self._hyde_rag,
            RAGStrategy.MULTI_QUERY: self._multi_query_rag,
            RAGStrategy.CONTEXTUAL_COMPRESSION: self._contextual_compression_rag,
            RAGStrategy.FUSION: self._fusion_rag,
            RAGStrategy.RECURSIVE: self._recursive_rag,
            RAGStrategy.HIERARCHICAL: self._hierarchical_rag
        }
    
    async def query(
        self,
        query: str,
        strategy: Optional[RAGStrategy] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a RAG query with specified strategy
        
        Args:
            query: The user query
            strategy: RAG strategy to use (defaults to config)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Dictionary containing answer and supporting documents
        """
        strategy = strategy or self.config.strategy
        
        if strategy not in self.strategies:
            raise RAGError(f"Unknown strategy: {strategy}")
        
        # Execute the appropriate strategy
        return await self.strategies[strategy](query, **kwargs)
    
    async def _standard_rag(self, query: str, **kwargs) -> Dict[str, Any]:
        """Standard RAG: Simple retrieval and generation"""
        # Retrieve relevant documents
        documents = await self._retrieve_documents(query, self.config.top_k)
        
        # Generate answer using retrieved context
        answer = await self._generate_answer(query, documents)
        
        return {
            "answer": answer,
            "documents": [doc.to_dict() for doc in documents],
            "strategy": "standard"
        }
    
    async def _hyde_rag(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        HyDE (Hypothetical Document Embeddings) RAG
        
        1. Generate a hypothetical answer
        2. Use the hypothetical answer for retrieval
        3. Generate final answer with retrieved documents
        """
        # Step 1: Generate hypothetical answer
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question as if you had perfect knowledge."},
            {"role": "user", "content": query}
        ]
        
        hypothetical_response = await self.llm_client.complete(
            messages,
            temperature=0.7,
            max_tokens=300
        )
        
        hypothetical_answer = hypothetical_response.content
        
        # Step 2: Retrieve using hypothetical answer
        documents = await self._retrieve_documents(
            hypothetical_answer,
            self.config.top_k * 2  # Get more documents for HyDE
        )
        
        # Step 3: Re-rank based on original query
        reranked_docs = await self._rerank_documents(query, documents)
        
        # Step 4: Generate final answer
        answer = await self._generate_answer(
            query,
            reranked_docs[:self.config.rerank_top_k]
        )
        
        return {
            "answer": answer,
            "documents": [doc.to_dict() for doc in reranked_docs[:self.config.rerank_top_k]],
            "strategy": "hyde",
            "hypothetical_answer": hypothetical_answer
        }
    
    async def _multi_query_rag(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Multi-Query RAG
        
        1. Generate multiple related queries
        2. Retrieve documents for each query
        3. Combine and deduplicate results
        4. Generate answer from combined context
        """
        # Step 1: Generate multiple queries
        messages = [
            {"role": "system", "content": "Generate 3 different versions of the following question that capture different aspects or phrasings. Return only the questions, one per line."},
            {"role": "user", "content": query}
        ]
        
        queries_response = await self.llm_client.complete(
            messages,
            temperature=0.8,
            max_tokens=200
        )
        
        queries = [query] + queries_response.content.strip().split('\n')
        queries = [q.strip() for q in queries if q.strip()][:4]  # Max 4 queries
        
        # Step 2: Retrieve for each query
        all_documents = []
        seen_ids = set()
        
        for q in queries:
            docs = await self._retrieve_documents(q, self.config.top_k)
            for doc in docs:
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    all_documents.append(doc)
        
        # Step 3: Re-rank combined results
        reranked_docs = await self._rerank_documents(query, all_documents)
        
        # Step 4: Generate answer
        answer = await self._generate_answer(
            query,
            reranked_docs[:self.config.rerank_top_k]
        )
        
        return {
            "answer": answer,
            "documents": [doc.to_dict() for doc in reranked_docs[:self.config.rerank_top_k]],
            "strategy": "multi_query",
            "queries_used": queries
        }
    
    async def _contextual_compression_rag(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Contextual Compression RAG
        
        1. Retrieve documents normally
        2. Extract only relevant portions from each document
        3. Generate answer from compressed context
        """
        # Step 1: Standard retrieval
        documents = await self._retrieve_documents(query, self.config.top_k)
        
        # Step 2: Compress each document
        compressed_docs = []
        
        for doc in documents:
            messages = [
                {"role": "system", "content": "Extract only the portions of the following text that are relevant to answering the question. If nothing is relevant, respond with 'NO_RELEVANT_CONTENT'."},
                {"role": "user", "content": f"Question: {query}\n\nText: {doc.content}"}
            ]
            
            compression_response = await self.llm_client.complete(
                messages,
                temperature=0.3,
                max_tokens=300
            )
            
            compressed_content = compression_response.content
            
            if compressed_content != "NO_RELEVANT_CONTENT":
                compressed_doc = Document(
                    id=doc.id,
                    content=compressed_content,
                    metadata={**doc.metadata, "original_length": len(doc.content)},
                    score=doc.score
                )
                compressed_docs.append(compressed_doc)
        
        # Step 3: Generate answer from compressed context
        answer = await self._generate_answer(query, compressed_docs)
        
        return {
            "answer": answer,
            "documents": [doc.to_dict() for doc in compressed_docs],
            "strategy": "contextual_compression",
            "compression_ratio": sum(len(d.content) for d in compressed_docs) / sum(len(d.content) for d in documents)
        }
    
    async def _fusion_rag(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Fusion RAG - Combines multiple retrieval methods
        
        1. Perform keyword-based retrieval
        2. Perform semantic retrieval
        3. Perform HyDE retrieval
        4. Fuse results with reciprocal rank fusion
        5. Generate answer from fused results
        """
        retrieval_methods = []
        
        # Method 1: Standard semantic retrieval
        semantic_docs = await self._retrieve_documents(query, self.config.top_k)
        retrieval_methods.append(("semantic", semantic_docs))
        
        # Method 2: Keyword-based retrieval (simulated)
        keyword_docs = await self._keyword_retrieve(query, self.config.top_k)
        retrieval_methods.append(("keyword", keyword_docs))
        
        # Method 3: HyDE retrieval
        messages = [
            {"role": "system", "content": "Answer this question briefly:"},
            {"role": "user", "content": query}
        ]
        hyde_response = await self.llm_client.complete(messages, temperature=0.7, max_tokens=150)
        hyde_docs = await self._retrieve_documents(hyde_response.content, self.config.top_k)
        retrieval_methods.append(("hyde", hyde_docs))
        
        # Reciprocal Rank Fusion
        fused_docs = self._reciprocal_rank_fusion(retrieval_methods)
        
        # Generate answer
        answer = await self._generate_answer(
            query,
            fused_docs[:self.config.rerank_top_k]
        )
        
        return {
            "answer": answer,
            "documents": [doc.to_dict() for doc in fused_docs[:self.config.rerank_top_k]],
            "strategy": "fusion",
            "retrieval_methods": [method[0] for method in retrieval_methods]
        }
    
    async def _recursive_rag(self, query: str, depth: int = 2, **kwargs) -> Dict[str, Any]:
        """
        Recursive RAG - Iteratively refines retrieval and answers
        
        1. Initial retrieval and answer generation
        2. Identify gaps or unclear points in the answer
        3. Generate follow-up queries for gaps
        4. Retrieve additional information
        5. Refine the answer
        """
        all_documents = []
        iterations = []
        
        current_query = query
        current_answer = ""
        
        for i in range(depth):
            # Retrieve documents
            docs = await self._retrieve_documents(current_query, self.config.top_k)
            all_documents.extend(docs)
            
            # Generate answer
            if i == 0:
                current_answer = await self._generate_answer(current_query, docs)
            else:
                # Refine previous answer with new information
                messages = [
                    {"role": "system", "content": "Refine and expand the previous answer with the new information provided."},
                    {"role": "user", "content": f"Original question: {query}\n\nPrevious answer: {current_answer}\n\nNew information: {self._format_documents(docs)}\n\nProvide an improved answer."}
                ]
                
                refinement_response = await self.llm_client.complete(messages)
                current_answer = refinement_response.content
            
            iterations.append({
                "query": current_query,
                "documents_retrieved": len(docs),
                "answer_preview": current_answer[:200] + "..."
            })
            
            # Check if we need more information
            if i < depth - 1:
                # Generate follow-up query
                messages = [
                    {"role": "system", "content": "Based on the answer provided, identify what additional information would make it more complete. Generate ONE specific follow-up question."},
                    {"role": "user", "content": f"Question: {query}\n\nCurrent answer: {current_answer}\n\nWhat additional information is needed?"}
                ]
                
                followup_response = await self.llm_client.complete(messages, temperature=0.8)
                current_query = followup_response.content.strip()
        
        # Deduplicate documents
        unique_docs = self._deduplicate_documents(all_documents)
        
        return {
            "answer": current_answer,
            "documents": [doc.to_dict() for doc in unique_docs[:self.config.top_k]],
            "strategy": "recursive",
            "iterations": iterations,
            "total_documents": len(unique_docs)
        }
    
    async def _hierarchical_rag(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Hierarchical RAG - Uses document summaries for initial retrieval
        
        1. Retrieve from document summaries
        2. Retrieve full documents based on summary matches
        3. Perform detailed retrieval within selected documents
        4. Generate answer from hierarchical context
        """
        # This is a simplified version - in practice, you'd have pre-computed summaries
        
        # Step 1: Retrieve at document level (simulated with larger chunks)
        coarse_docs = await self._retrieve_documents(query, self.config.top_k * 2)
        
        # Step 2: For each relevant document, retrieve finer chunks
        fine_documents = []
        
        for doc in coarse_docs[:self.config.top_k]:
            # Simulate retrieving smaller chunks from the same document
            # In practice, this would query chunks with doc.id as a filter
            chunks = await self._retrieve_document_chunks(doc.id, query, 3)
            fine_documents.extend(chunks)
        
        # Step 3: Re-rank all fine-grained chunks
        reranked_docs = await self._rerank_documents(query, fine_documents)
        
        # Step 4: Generate answer
        answer = await self._generate_answer(
            query,
            reranked_docs[:self.config.rerank_top_k]
        )
        
        return {
            "answer": answer,
            "documents": [doc.to_dict() for doc in reranked_docs[:self.config.rerank_top_k]],
            "strategy": "hierarchical",
            "coarse_documents": len(coarse_docs),
            "fine_documents": len(fine_documents)
        }
    
    # Helper methods
    
    @cache_result(ttl=3600)
    async def _retrieve_documents(self, query: str, top_k: int) -> List[Document]:
        """Retrieve documents from vector store"""
        results = await self.vector_store.search(query, top_k)
        
        documents = []
        for result in results:
            doc = Document(
                id=result.get("id", ""),
                content=result.get("content", ""),
                metadata=result.get("metadata", {}),
                score=result.get("score", 0.0)
            )
            documents.append(doc)
        
        return documents
    
    async def _keyword_retrieve(self, query: str, top_k: int) -> List[Document]:
        """Simulate keyword-based retrieval"""
        # In practice, this would use BM25 or similar
        # For now, we'll use the vector store with a modified query
        keywords = query.lower().split()
        keyword_query = " OR ".join(keywords)
        return await self._retrieve_documents(keyword_query, top_k)
    
    async def _retrieve_document_chunks(
        self,
        doc_id: str,
        query: str,
        top_k: int
    ) -> List[Document]:
        """Retrieve chunks from a specific document"""
        # In practice, this would filter by doc_id
        # For now, simulate with regular retrieval
        docs = await self._retrieve_documents(query, top_k)
        # Add parent_doc_id to metadata
        for doc in docs:
            doc.metadata["parent_doc_id"] = doc_id
        return docs
    
    async def _rerank_documents(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        """Re-rank documents based on relevance to query"""
        if not documents:
            return []
        
        # Use LLM for re-ranking
        reranked = []
        
        for doc in documents:
            messages = [
                {"role": "system", "content": "Rate the relevance of the following text to the question on a scale of 0-10. Respond with only a number."},
                {"role": "user", "content": f"Question: {query}\n\nText: {doc.content[:500]}"}
            ]
            
            score_response = await self.llm_client.complete(
                messages,
                temperature=0.1,
                max_tokens=10
            )
            
            try:
                score = float(score_response.content.strip())
                doc.score = score / 10.0
            except:
                doc.score = 0.5  # Default score if parsing fails
            
            reranked.append(doc)
        
        # Sort by score
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked
    
    def _reciprocal_rank_fusion(
        self,
        retrieval_methods: List[Tuple[str, List[Document]]]
    ) -> List[Document]:
        """Perform reciprocal rank fusion on multiple retrieval results"""
        k = 60  # Constant for RRF
        doc_scores = defaultdict(float)
        doc_objects = {}
        
        for method_name, documents in retrieval_methods:
            for rank, doc in enumerate(documents):
                # RRF formula
                score = 1.0 / (k + rank + 1)
                doc_scores[doc.id] += score
                doc_objects[doc.id] = doc
        
        # Sort by fused score
        sorted_ids = sorted(doc_scores.keys(), key=doc_scores.get, reverse=True)
        
        fused_documents = []
        for doc_id in sorted_ids:
            doc = doc_objects[doc_id]
            doc.score = doc_scores[doc_id]
            fused_documents.append(doc)
        
        return fused_documents
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents"""
        seen_ids = set()
        unique_docs = []
        
        for doc in documents:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                unique_docs.append(doc)
        
        return unique_docs
    
    async def _generate_answer(
        self,
        query: str,
        documents: List[Document]
    ) -> str:
        """Generate answer using retrieved documents"""
        if not documents:
            return "I couldn't find relevant information to answer your question."
        
        context = self._format_documents(documents)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain enough information, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        response = await self.llm_client.complete(
            messages,
            temperature=self.config.temperature,
            max_tokens=500
        )
        
        return response.content
    
    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents for inclusion in prompt"""
        formatted = []
        for i, doc in enumerate(documents):
            formatted.append(f"[Document {i+1}]\n{doc.content}\n")
        return "\n".join(formatted)


# Convenience functions

async def create_advanced_rag(
    strategy: str = "standard",
    vector_store: Optional[VectorStore] = None,
    **kwargs
) -> AdvancedRAG:
    """
    Create an Advanced RAG instance
    
    Example:
        rag = await create_advanced_rag("hyde", llm_provider="openai")
        result = await rag.query("What is quantum computing?")
    """
    config = RAGConfig(
        strategy=RAGStrategy(strategy),
        vector_store=vector_store,
        **kwargs
    )
    
    return AdvancedRAG(config)


async def compare_rag_strategies(
    query: str,
    strategies: List[str],
    vector_store: Optional[VectorStore] = None
) -> Dict[str, Any]:
    """
    Compare multiple RAG strategies on the same query
    
    Returns comparative results and performance metrics
    """
    results = {}
    
    for strategy in strategies:
        rag = await create_advanced_rag(strategy, vector_store)
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await rag.query(query)
            end_time = asyncio.get_event_loop().time()
            
            results[strategy] = {
                "answer": result["answer"],
                "num_documents": len(result.get("documents", [])),
                "execution_time": end_time - start_time,
                "strategy_details": {k: v for k, v in result.items() if k not in ["answer", "documents"]}
            }
        except Exception as e:
            results[strategy] = {
                "error": str(e),
                "execution_time": 0
            }
    
    return results
