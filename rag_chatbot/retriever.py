"""
Retriever Module
Handles similarity search to find relevant context for questions.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with relevance info."""
    text: str
    source: str
    distance: float
    chunk_id: int
    relevance_score: float  # 1 - distance for cosine
    
    def __repr__(self):
        return f"RetrievedChunk(source='{self.source}', score={self.relevance_score:.3f})"


class Retriever:
    """
    Retrieves relevant context from the vector store based on query similarity.
    """
    
    def __init__(self, vector_store, embedder):
        """
        Initialize the retriever.
        
        Args:
            vector_store: VectorStore instance
            embedder: Embedder instance
        """
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_relevance: float = 0.0,
        source_filter: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: User's question
            top_k: Maximum number of chunks to retrieve
            min_relevance: Minimum relevance score (0-1)
            source_filter: Optional filter by source name
            
        Returns:
            List of RetrievedChunk objects, sorted by relevance
        """
        # Embed the query
        query_embedding = self.embedder.embed(query)[0].tolist()
        
        # Build filter if needed
        where = None
        if source_filter:
            where = {"source": source_filter}
        
        # Query the vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where
        )
        
        # Convert to RetrievedChunk objects
        chunks = []
        for doc, meta, dist in zip(
            results["documents"],
            results["metadatas"],
            results["distances"]
        ):
            # Convert distance to relevance score (1 - distance for cosine)
            relevance = 1 - dist
            
            if relevance >= min_relevance:
                chunks.append(RetrievedChunk(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    distance=dist,
                    chunk_id=meta.get("chunk_id", 0),
                    relevance_score=relevance
                ))
        
        return chunks
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5
    ) -> str:
        """
        Retrieve relevant chunks and format them as context for the LLM.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Formatted context string
        """
        chunks = self.retrieve(query, top_k=top_k)
        
        if not chunks:
            return "No relevant context found in the knowledge base."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk.source} (relevance: {chunk.relevance_score:.2f})]\n"
                f"{chunk.text}"
            )
        
        return "\n\n".join(context_parts)
    
    def get_sources(self, query: str, top_k: int = 5) -> List[str]:
        """
        Get the source documents for retrieved chunks.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            List of unique source names
        """
        chunks = self.retrieve(query, top_k=top_k)
        return list(set(chunk.source for chunk in chunks))


if __name__ == "__main__":
    print("Retriever module loaded. Use with VectorStore and Embedder.")
