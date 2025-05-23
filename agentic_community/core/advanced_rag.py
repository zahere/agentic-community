"""
Advanced RAG (Retrieval-Augmented Generation) techniques.

Implements state-of-the-art RAG strategies including:
- HyDE (Hypothetical Document Embeddings)
- Multi-Query Generation
- Hybrid Search (Vector + Keyword)
- Re-ranking Strategies
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

from ..core.vector_store import VectorStore, Document, SearchResult
from ..core.llm_providers import LLMRouter, LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG strategies."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    rerank_top_k: int = 3
    use_hyde: bool = True
    use_multi_query: bool = True
    use_hybrid_search: bool = True
    hyde_temperature: float = 0.7
    multi_query_count: int = 3
    relevance_threshold: float = 0.7


@dataclass
class QueryResult:
    """Enhanced query result with metadata."""
    original_query: str
    expanded_queries: List[str] = field(default_factory=list)
    hypothetical_answer: Optional[str] = None
    retrieved_documents: List[Document] = field(default_factory=list)
    reranked_documents: List[Document] = field(default_factory=list)
    final_context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class HyDEStrategy:
    """
    Hypothetical Document Embeddings (HyDE) strategy.
    
    Generates a hypothetical answer to the query and uses it
    for more effective semantic search.
    """
    
    def __init__(
        self, 
        llm_router: LLMRouter,
        embedding_function: Callable[[str], List[float]]
    ):
        self.llm_router = llm_router
        self.embed = embedding_function
        
    async def generate_hypothetical_answer(self, query: str) -> str:
        """Generate a hypothetical answer for the query."""
        prompt = f"""Generate a comprehensive, factual answer to this question. 
Even if you're not certain, provide a detailed response that would likely contain the correct information.

Question: {query}

Detailed Answer:"""
        
        response = await self.llm_router.complete(prompt, temperature=0.7)
        return response.content


class MultiQueryStrategy:
    """
    Multi-Query Generation strategy.
    
    Generates multiple variations of the query to improve recall.
    """
    
    def __init__(
        self,
        llm_router: LLMRouter,
        num_queries: int = 3
    ):
        self.llm_router = llm_router
        self.num_queries = num_queries
        
    async def generate_queries(self, original_query: str) -> List[str]:
        """Generate multiple query variations."""
        prompt = f"""Generate {self.num_queries} different variations of this question.
Make them diverse but semantically similar.

Original: {original_query}

Variations:
1."""
        
        response = await self.llm_router.complete(prompt)
        
        # Parse the response
        lines = response.content.strip().split('\n')
        queries = [original_query]  # Include original
        
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and '.' in line:
                query = line.split('.', 1)[1].strip()
                if query:
                    queries.append(query)
                    
        return queries[:self.num_queries + 1]


class HybridSearchStrategy:
    """
    Hybrid search combining vector similarity and keyword matching.
    """
    
    def __init__(
        self,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7
    ):
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        
    def keyword_score(self, query: str, document: str) -> float:
        """Calculate keyword-based relevance score."""
        query_tokens = set(query.lower().split())
        doc_tokens = set(document.lower().split())
        
        if not query_tokens:
            return 0.0
            
        # Jaccard similarity
        intersection = query_tokens.intersection(doc_tokens)
        union = query_tokens.union(doc_tokens)
        
        return len(intersection) / len(union) if union else 0.0
        
    def apply_hybrid_scoring(self, query: str, documents: List[Document]) -> List[Document]:
        """Apply hybrid scoring to documents."""
        for doc in documents:
            keyword_score = self.keyword_score(query, doc.content)
            vector_score = doc.metadata.get("vector_score", 0.5)
            
            combined_score = (
                self.vector_weight * vector_score +
                self.keyword_weight * keyword_score
            )
            
            doc.metadata["hybrid_score"] = combined_score
            doc.metadata["keyword_score"] = keyword_score
            
        # Re-sort by hybrid score
        documents.sort(
            key=lambda d: d.metadata.get("hybrid_score", 0),
            reverse=True
        )
        
        return documents


class CrossEncoderReranker:
    """
    Re-ranking strategy using LLM-based scoring.
    """
    
    def __init__(
        self,
        llm_router: LLMRouter,
        top_k: int = 3
    ):
        self.llm_router = llm_router
        self.top_k = top_k
        
    async def score_relevance(self, query: str, document: str) -> float:
        """Score the relevance of a document to a query."""
        prompt = f"""Rate relevance (0-10): Query: {query}
Document: {document[:300]}...
Score:"""
        
        response = await self.llm_router.complete(prompt, max_tokens=10)
        
        try:
            score = float(response.content.strip())
            return min(max(score / 10.0, 0.0), 1.0)
        except:
            return 0.5
            
    async def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Re-rank documents based on relevance scores."""
        scores = await asyncio.gather(*[
            self.score_relevance(query, doc.content)
            for doc in documents
        ])
        
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = score
            
        documents.sort(
            key=lambda d: d.metadata.get("rerank_score", 0),
            reverse=True
        )
        
        return documents[:self.top_k]


class AdvancedRAGEngine:
    """
    Advanced RAG engine combining multiple strategies.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_router: LLMRouter,
        embedding_function: Callable[[str], List[float]],
        config: Optional[RAGConfig] = None
    ):
        self.vector_store = vector_store
        self.llm_router = llm_router
        self.embed = embedding_function
        self.config = config or RAGConfig()
        
        # Initialize strategies
        self.hyde = HyDEStrategy(llm_router, embedding_function)
        self.multi_query = MultiQueryStrategy(llm_router, config.multi_query_count)
        self.hybrid_search = HybridSearchStrategy()
        self.reranker = CrossEncoderReranker(llm_router, config.rerank_top_k)
        
    async def retrieve(self, query: str) -> QueryResult:
        """
        Perform advanced retrieval using multiple strategies.
        """
        result = QueryResult(original_query=query)
        
        # Step 1: Multi-query generation
        if self.config.use_multi_query:
            result.expanded_queries = await self.multi_query.generate_queries(query)
        else:
            result.expanded_queries = [query]
            
        # Step 2: HyDE generation
        if self.config.use_hyde:
            result.hypothetical_answer = await self.hyde.generate_hypothetical_answer(query)
            
        # Step 3: Retrieve documents
        all_documents = []
        seen_ids = set()
        
        for expanded_query in result.expanded_queries:
            # Use HyDE embedding if available
            if result.hypothetical_answer and self.config.use_hyde:
                search_embedding = self.embed(result.hypothetical_answer)
            else:
                search_embedding = self.embed(expanded_query)
                
            # Vector search
            search_results = self.vector_store.search(
                search_embedding,
                k=self.config.top_k * 2
            )
            
            for sr in search_results:
                if sr.document.id not in seen_ids:
                    sr.document.metadata["vector_score"] = sr.score
                    all_documents.append(sr.document)
                    seen_ids.add(sr.document.id)
                    
        result.retrieved_documents = all_documents
        
        # Step 4: Hybrid search scoring
        if self.config.use_hybrid_search and all_documents:
            all_documents = self.hybrid_search.apply_hybrid_scoring(query, all_documents)
            
        # Step 5: Re-ranking
        if all_documents:
            result.reranked_documents = await self.reranker.rerank(query, all_documents)
        else:
            result.reranked_documents = []
            
        # Build final context
        context_parts = []
        for i, doc in enumerate(result.reranked_documents):
            relevance = doc.metadata.get("rerank_score", 0)
            context_parts.append(f"[Doc {i+1} - Score: {relevance:.2f}]\n{doc.content}")
            
        result.final_context = "\n\n".join(context_parts)
        
        return result
        
    async def query(self, query: str) -> str:
        """
        Query the RAG system and generate an answer.
        """
        result = await self.retrieve(query)
        
        prompt = f"""Answer based on the context provided.

Context:
{result.final_context}

Question: {query}

Answer:"""
        
        response = await self.llm_router.complete(prompt)
        return response.content
