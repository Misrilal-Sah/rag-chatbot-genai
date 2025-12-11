"""
RAG Chatbot Module
Main module that orchestrates the complete RAG pipeline.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .pdf_loader import load_all_pdfs_from_directory
from .audio_transcriber import load_all_audio_from_directory
from .chunker import TextChunker, TextChunk, create_chunks
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever, RetrievedChunk
from .generator import Generator, LightGenerator


@dataclass
class RAGResponse:
    """Represents a response from the RAG chatbot."""
    question: str
    answer: str
    sources: List[RetrievedChunk]
    context_used: str


class RAGChatbot:
    """
    Complete RAG (Retrieval-Augmented Generation) Chatbot.
    
    This class orchestrates the entire pipeline:
    1. Load documents (PDFs and audio)
    2. Chunk the text
    3. Generate embeddings
    4. Store in vector database
    5. Retrieve relevant context
    6. Generate responses with LLM
    """
    
    def __init__(
        self,
        data_directory: str,
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_knowledge_base",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_light_llm: bool = True,  # Use lighter model by default
        whisper_model: str = "base"
    ):
        """
        Initialize the RAG chatbot.
        
        Args:
            data_directory: Directory containing PDFs and audio files
            persist_directory: Directory to persist the vector database
            collection_name: Name for the ChromaDB collection
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Sentence-transformer model name
            use_light_llm: If True, use lighter/faster LLM
            whisper_model: Whisper model size for transcription
        """
        self.data_directory = Path(data_directory)
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.whisper_model = whisper_model
        
        print("=" * 60)
        print("Initializing RAG Chatbot")
        print("=" * 60)
        
        # Initialize components
        print("\n[1/4] Loading embedding model...")
        self.embedder = Embedder(model_name=embedding_model)
        
        print("\n[2/4] Initializing vector store...")
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        print("\n[3/4] Setting up retriever...")
        self.retriever = Retriever(self.vector_store, self.embedder)
        
        print("\n[4/4] Loading LLM...")
        if use_light_llm:
            self.generator = LightGenerator()
        else:
            self.generator = Generator()
        
        self.is_indexed = self.vector_store.count() > 0
        
        print("\n" + "=" * 60)
        print("✓ RAG Chatbot initialized!")
        if self.is_indexed:
            print(f"  Knowledge base contains {self.vector_store.count()} chunks")
        else:
            print("  Knowledge base is empty. Run index_documents() to populate.")
        print("=" * 60)
    
    def index_documents(self, force_reindex: bool = False) -> int:
        """
        Load, process, and index all documents from the data directory.
        
        Args:
            force_reindex: If True, clear existing index and rebuild
            
        Returns:
            Number of chunks indexed
        """
        print("\n" + "=" * 60)
        print("Indexing Documents")
        print("=" * 60)
        
        if force_reindex:
            print("Clearing existing index...")
            self.vector_store.clear()
        elif self.is_indexed:
            print(f"Index already contains {self.vector_store.count()} chunks.")
            print("Use force_reindex=True to rebuild.")
            return self.vector_store.count()
        
        # Step 1: Load PDFs
        print("\n[Step 1] Loading PDFs...")
        pdf_texts = load_all_pdfs_from_directory(str(self.data_directory))
        
        # Step 2: Transcribe audio (this may take a while)
        print("\n[Step 2] Transcribing audio/video files...")
        print("(This may take 10-20 minutes for large files on first run)")
        audio_texts = load_all_audio_from_directory(
            str(self.data_directory),
            model_name=self.whisper_model
        )
        
        # Combine all documents
        all_documents = {**pdf_texts, **audio_texts}
        
        if not all_documents:
            print("No documents found to index!")
            return 0
        
        print(f"\nLoaded {len(all_documents)} document(s):")
        for name, text in all_documents.items():
            print(f"  - {name}: {len(text)} characters")
        
        # Step 3: Chunk the documents
        print("\n[Step 3] Chunking documents...")
        chunks = create_chunks(
            all_documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Step 4: Generate embeddings
        print("\n[Step 4] Generating embeddings...")
        embeddings = self.embedder.embed_chunks(chunks)
        
        # Step 5: Store in vector database
        print("\n[Step 5] Storing in vector database...")
        self.vector_store.add_chunks(chunks, embeddings)
        
        self.is_indexed = True
        
        print("\n" + "=" * 60)
        print(f"✓ Indexing complete! {len(chunks)} chunks stored.")
        print("=" * 60)
        
        return len(chunks)
    
    def ask(
        self,
        question: str,
        top_k: int = 5,
        verbose: bool = False
    ) -> RAGResponse:
        """
        Ask a question and get a response based on the knowledge base.
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            verbose: If True, print retrieval details
            
        Returns:
            RAGResponse object with answer and sources
        """
        if not self.is_indexed:
            return RAGResponse(
                question=question,
                answer="Knowledge base is empty. Please run index_documents() first.",
                sources=[],
                context_used=""
            )
        
        # Retrieve relevant context
        chunks = self.retriever.retrieve(question, top_k=top_k)
        context = self.retriever.retrieve_with_context(question, top_k=top_k)
        
        if verbose:
            print(f"\nRetrieved {len(chunks)} relevant chunks:")
            for chunk in chunks:
                print(f"  - {chunk.source}: score={chunk.relevance_score:.3f}")
        
        # Generate response
        answer = self.generator.generate_rag_response(question, context)
        
        return RAGResponse(
            question=question,
            answer=answer,
            sources=chunks,
            context_used=context
        )
    
    def chat(self, question: str) -> str:
        """
        Simple chat interface - just returns the answer string.
        
        Args:
            question: User's question
            
        Returns:
            Answer string
        """
        response = self.ask(question)
        return response.answer
    
    def interactive_chat(self):
        """
        Start an interactive chat session.
        Type 'quit' or 'exit' to end the session.
        """
        print("\n" + "=" * 60)
        print("RAG Chatbot - Interactive Mode")
        print("=" * 60)
        print("Ask questions about the knowledge base.")
        print("Type 'quit' or 'exit' to end the session.\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                response = self.ask(question, verbose=True)
                print(f"\nBot: {response.answer}")
                print(f"\n[Sources: {', '.join(set(c.source for c in response.sources))}]\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


def create_chatbot(
    data_directory: str = "./Data",
    **kwargs
) -> RAGChatbot:
    """
    Factory function to create a RAG chatbot.
    
    Args:
        data_directory: Path to the data directory
        **kwargs: Additional arguments for RAGChatbot
        
    Returns:
        Initialized RAGChatbot instance
    """
    return RAGChatbot(data_directory=data_directory, **kwargs)


if __name__ == "__main__":
    # Quick test
    print("RAG Chatbot module loaded successfully!")
    print("Use create_chatbot() to create an instance.")
