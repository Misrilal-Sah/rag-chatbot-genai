"""
Text Chunking Module
Splits text into smaller, semantically meaningful chunks for embedding.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    source: str
    chunk_id: int
    start_char: int
    end_char: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char
        }


class TextChunker:
    """
    Splits text into overlapping chunks for better context preservation.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            separators: List of separators to split on (in order of preference)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def _split_text(self, text: str, separator: str) -> List[str]:
        """Split text by separator, keeping the separator at the end of each chunk."""
        if separator:
            splits = text.split(separator)
            # Add separator back to each split (except the last)
            return [s + separator if i < len(splits) - 1 else s 
                    for i, s in enumerate(splits) if s]
        else:
            return list(text)
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks of appropriate size."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split)
            
            # If adding this split would exceed chunk size
            if current_length + split_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append("".join(current_chunk))
                
                # Start new chunk with overlap from previous
                overlap_text = "".join(current_chunk)
                if len(overlap_text) > self.chunk_overlap:
                    overlap_text = overlap_text[-self.chunk_overlap:]
                current_chunk = [overlap_text] if overlap_text.strip() else []
                current_length = len(overlap_text)
            
            current_chunk.append(split)
            current_length += split_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        return chunks
    
    def chunk_text(self, text: str, source: str = "unknown") -> List[TextChunk]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            source: Source identifier for the text
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Clean the text
        text = text.strip()
        
        # Try each separator in order
        chunks_text = [text]
        
        for separator in self.separators:
            new_chunks = []
            for chunk in chunks_text:
                if len(chunk) > self.chunk_size:
                    splits = self._split_text(chunk, separator)
                    merged = self._merge_splits(splits, separator)
                    new_chunks.extend(merged)
                else:
                    new_chunks.append(chunk)
            chunks_text = new_chunks
        
        # Create TextChunk objects with metadata
        text_chunks = []
        current_pos = 0
        
        for i, chunk_text in enumerate(chunks_text):
            if chunk_text.strip():  # Skip empty chunks
                # Find the actual position in original text
                start_pos = text.find(chunk_text[:50], current_pos)
                if start_pos == -1:
                    start_pos = current_pos
                
                text_chunks.append(TextChunk(
                    text=chunk_text.strip(),
                    source=source,
                    chunk_id=i,
                    start_char=start_pos,
                    end_char=start_pos + len(chunk_text)
                ))
                current_pos = start_pos + len(chunk_text) - self.chunk_overlap
        
        return text_chunks
    
    def chunk_documents(self, documents: Dict[str, str]) -> List[TextChunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: Dictionary mapping source names to text content
            
        Returns:
            List of all TextChunk objects
        """
        all_chunks = []
        
        for source, text in documents.items():
            chunks = self.chunk_text(text, source)
            all_chunks.extend(chunks)
            print(f"✓ Created {len(chunks)} chunks from: {source}")
        
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


def create_chunks(
    documents: Dict[str, str],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[TextChunk]:
    """
    Convenience function to chunk multiple documents.
    
    Args:
        documents: Dictionary mapping source names to text content
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of TextChunk objects
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_documents(documents)


if __name__ == "__main__":
    # Test the chunker
    sample_text = """
    This is a sample document about RAG systems. RAG stands for Retrieval-Augmented Generation.
    
    It combines the power of retrieval systems with generative AI models.
    
    The main components are:
    1. A knowledge base with documents
    2. An embedding model to convert text to vectors
    3. A vector database for similarity search
    4. An LLM for generating responses
    
    This approach helps reduce hallucinations and provides more accurate answers.
    """
    
    chunker = TextChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk_text(sample_text, "sample.txt")
    
    print(f"Created {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"Chunk {chunk.chunk_id}: {len(chunk.text)} chars")
        print(f"  Text: {chunk.text[:100]}...")
        print()
