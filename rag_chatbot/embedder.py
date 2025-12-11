"""
Embedding Module
Uses sentence-transformers to convert text into vector embeddings.
"""

from typing import List, Union
import numpy as np


class Embedder:
    """
    Generates embeddings using sentence-transformers (HuggingFace).
    Uses the all-MiniLM-L6-v2 model by default - fast and good quality.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a sentence-transformer model.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Loaded model with {self.embedding_dim} dimensions")
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            text: Single string or list of strings to embed
            
        Returns:
            Numpy array of embeddings (2D array for multiple texts)
        """
        if isinstance(text, str):
            text = [text]
        
        embeddings = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=len(text) > 10
        )
        
        return embeddings
    
    def embed_chunks(self, chunks: List) -> List[np.ndarray]:
        """
        Embed a list of TextChunk objects.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            List of embedding arrays
        """
        texts = [chunk.text for chunk in chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embed(texts)
        print(f"✓ Generated {len(embeddings)} embeddings")
        return embeddings.tolist()


# Global embedder instance (lazy loaded)
_embedder = None


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None or _embedder.model_name != model_name:
        _embedder = Embedder(model_name)
    return _embedder


def embed_texts(texts: Union[str, List[str]], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Convenience function to embed texts.
    
    Args:
        texts: Text or list of texts to embed
        model_name: Embedding model name
        
    Returns:
        Numpy array of embeddings
    """
    embedder = get_embedder(model_name)
    return embedder.embed(texts)


if __name__ == "__main__":
    # Test the embedder
    test_texts = [
        "What is RAG?",
        "Retrieval-Augmented Generation combines search with LLMs",
        "Vector databases store embeddings for similarity search"
    ]
    
    embedder = Embedder()
    embeddings = embedder.embed(test_texts)
    
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"First embedding (first 10 values): {embeddings[0][:10]}")
    
    # Test similarity
    from numpy.linalg import norm
    similarity = np.dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
    print(f"\nSimilarity between query and RAG definition: {similarity:.4f}")
