"""
Advanced RAG Agent Example

This example demonstrates how to build a knowledge-augmented agent using:
- MCP (Model Context Protocol) for standardized tool interfaces
- Qdrant vector database for semantic search
- RAG (Retrieval-Augmented Generation) for enhanced responses
"""

import os
import asyncio
from typing import List, Dict, Any

# Mock imports for demonstration (replace with actual implementations)
from agentic_community import SimpleAgent
from agentic_community.core.mcp import MCPServer, MCPClient, create_mcp_server
from agentic_community.core.vector_store import (
    QdrantVectorStore, SemanticMemory, RAGTool, KnowledgeBase
)
from agentic_community.tools import SearchTool, TextTool


# Mock embedding function (replace with OpenAI embeddings in production)
def get_embeddings(text: str) -> List[float]:
    """
    Get embeddings for text.
    In production, use:
    - OpenAI's text-embedding-ada-002
    - Cohere's embed-english-v3.0
    - Or other embedding models
    """
    # This is a mock - see vector_store.py for the implementation
    from agentic_community.core.vector_store import mock_embedding_function
    return mock_embedding_function(text)


class RAGAgent(SimpleAgent):
    """Enhanced agent with RAG capabilities and MCP support."""
    
    def __init__(self, name: str = "RAG Assistant"):
        super().__init__(name)
        
        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            collection_name="agent_knowledge",
            host="localhost",
            port=6333
        )
        
        # Initialize semantic memory
        self.semantic_memory = SemanticMemory(
            vector_store=self.vector_store,
            embedding_function=get_embeddings
        )
        
        # Initialize knowledge base
        self.knowledge_base = KnowledgeBase(
            vector_store=self.vector_store,
            embedding_function=get_embeddings
        )
        
        # Add RAG tool
        self.rag_tool = RAGTool(
            vector_store=self.vector_store,
            embedding_function=get_embeddings
        )
        self.add_tool(self.rag_tool)
        
        # Initialize MCP server
        self.mcp_server = create_mcp_server()
        self._register_mcp_tools()
        
    def _register_mcp_tools(self):
        """Register tools with MCP server for standardized access."""
        for tool in self.tools:
            self.mcp_server.register_tool(tool)
            
    async def process_with_context(self, query: str) -> str:
        """Process query with RAG context enhancement."""
        # Step 1: Retrieve relevant context
        relevant_docs = self.semantic_memory.recall(query, k=3)
        
        # Step 2: Build enhanced prompt
        context = "\n".join([doc.content for doc in relevant_docs])
        enhanced_prompt = f"""
        Context from knowledge base:
        {context}
        
        User query: {query}
        
        Please provide a comprehensive answer using the context above.
        """
        
        # Step 3: Process with enhanced context
        response = await self.arun(enhanced_prompt)
        
        # Step 4: Store the interaction in memory
        self.semantic_memory.remember(
            f"Q: {query}\nA: {response}",
            metadata={"type": "conversation"}
        )
        
        return response
        
    def add_knowledge(self, documents: List[str], source: str):
        """Add documents to the knowledge base."""
        chunks_added = self.knowledge_base.add_knowledge(
            documents=documents,
            source=source,
            chunk_size=500,
            chunk_overlap=50
        )
        print(f"Added {chunks_added} chunks from {source}")


async def mcp_tool_discovery_example():
    """Example of tool discovery using MCP."""
    print("=== MCP Tool Discovery Example ===\n")
    
    # Create MCP client
    client = MCPClient()
    
    # Create and register some tools
    server = client.local_server
    server.register_tool(SearchTool(), "web_search")
    server.register_tool(TextTool(), "text_processor")
    
    # Discover available tools
    tools = await client.list_tools()
    
    print("Available tools via MCP:")
    for tool in tools:
        print(f"- {tool['name']}: {tool['schema']['description']}")
        
    # Execute a tool via MCP
    result = await client.execute_tool(
        "web_search",
        {"query": "latest AI agent frameworks"}
    )
    print(f"\nSearch result: {result}")


async def rag_agent_example():
    """Example of using RAG-enhanced agent."""
    print("\n=== RAG Agent Example ===\n")
    
    # Initialize RAG agent
    agent = RAGAgent("Knowledge Assistant")
    
    # Add some knowledge to the agent
    knowledge_docs = [
        """
        The Agentic Framework is an open-source platform for building AI agents.
        It supports multiple tools, memory systems, and now includes MCP protocol
        support for standardized tool interfaces.
        """,
        """
        Qdrant is a vector database designed for similarity search and neural network
        applications. It provides high-performance vector similarity search with
        filtering capabilities.
        """,
        """
        RAG (Retrieval-Augmented Generation) enhances LLM responses by retrieving
        relevant information from a knowledge base before generating answers.
        This approach reduces hallucinations and improves accuracy.
        """
    ]
    
    agent.add_knowledge(knowledge_docs, source="framework_docs")
    
    # Ask questions that benefit from RAG
    questions = [
        "What is the Agentic Framework?",
        "How does Qdrant help with AI applications?",
        "What are the benefits of RAG?"
    ]
    
    for question in questions:
        print(f"Q: {question}")
        response = await agent.process_with_context(question)
        print(f"A: {response}\n")


async def semantic_memory_example():
    """Example of semantic memory usage."""
    print("\n=== Semantic Memory Example ===\n")
    
    # Create vector store and semantic memory
    vector_store = QdrantVectorStore(
        collection_name="conversation_memory",
        vector_size=1536  # OpenAI embedding size
    )
    
    memory = SemanticMemory(
        vector_store=vector_store,
        embedding_function=get_embeddings
    )
    
    # Store some memories
    memories = [
        "User prefers technical explanations with code examples",
        "User is building a chatbot for customer service",
        "User mentioned they use Python and FastAPI",
        "User asked about rate limiting yesterday"
    ]
    
    print("Storing memories...")
    for mem in memories:
        memory.remember(mem, metadata={"type": "user_preference"})
        
    # Recall relevant memories
    query = "How should I implement the API?"
    print(f"\nQuery: {query}")
    
    relevant_memories = memory.recall(query, k=2)
    print("\nRelevant memories:")
    for mem in relevant_memories:
        print(f"- {mem.content}")


def knowledge_base_integration_example():
    """Example of integrating external knowledge bases."""
    print("\n=== Knowledge Base Integration Example ===\n")
    
    # Create knowledge base
    kb = KnowledgeBase(
        vector_store=QdrantVectorStore(collection_name="external_kb"),
        embedding_function=get_embeddings
    )
    
    # Simulate loading documentation
    api_docs = [
        """
        POST /api/agents/create
        Creates a new agent instance.
        Required parameters: name (string), tools (array of tool names)
        Returns: agent_id (string), status (string)
        """,
        """
        GET /api/agents/{agent_id}/status
        Retrieves the current status of an agent.
        Returns: status (running|idle|error), last_activity (timestamp)
        """,
        """
        POST /api/agents/{agent_id}/execute
        Executes a task with the specified agent.
        Required parameters: task (string), context (optional object)
        Returns: result (string), execution_time (float)
        """
    ]
    
    # Add to knowledge base
    chunks = kb.add_knowledge(api_docs, source="api_documentation")
    print(f"Added {chunks} chunks to knowledge base")
    
    # Query the knowledge base
    queries = [
        "How do I create an agent?",
        "What parameters does the execute endpoint need?",
        "How can I check agent status?"
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        results = kb.query(q, k=1)
        if results:
            print(f"Answer: {results[0].content}")


async def main():
    """Run all examples."""
    print("🚀 Advanced RAG Agent Examples\n")
    
    # Run MCP tool discovery
    await mcp_tool_discovery_example()
    
    # Run RAG agent example
    await rag_agent_example()
    
    # Run semantic memory example
    await semantic_memory_example()
    
    # Run knowledge base integration
    knowledge_base_integration_example()
    
    print("\n✨ All examples completed!")


if __name__ == "__main__":
    # Note: In production, ensure Qdrant is running
    # docker run -p 6333:6333 qdrant/qdrant
    
    asyncio.run(main())
