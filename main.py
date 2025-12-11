"""
RAG Chatbot - Main Entry Point

This script initializes and runs the RAG chatbot pipeline.
It processes PDFs and audio files from the Data directory,
builds a vector index, and provides a chat interface.

Usage:
    python main.py                  # Index and start interactive chat
    python main.py --reindex        # Force reindex all documents
    python main.py --test           # Run test questions only
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag_chatbot.chatbot import RAGChatbot


def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot for lecture materials")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./Data",
        help="Directory containing PDFs and audio files"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindex all documents"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test questions instead of interactive mode"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for audio transcription"
    )
    
    args = parser.parse_args()
    
    # Verify data directory exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Please ensure your PDFs and audio files are in the Data folder.")
        sys.exit(1)
    
    # Initialize chatbot
    print("\n" + "=" * 70)
    print("  RAG Chatbot - GenAI Lecture Knowledge Base")
    print("=" * 70)
    
    chatbot = RAGChatbot(
        data_directory=args.data_dir,
        persist_directory="./chroma_db",
        collection_name="genai_lectures",
        chunk_size=500,
        chunk_overlap=50,
        use_light_llm=True,  # Use faster model
        whisper_model=args.whisper_model
    )
    
    # Index documents if needed
    chatbot.index_documents(force_reindex=args.reindex)
    
    if args.test:
        # Run test questions
        run_test_questions(chatbot)
    else:
        # Start interactive chat
        chatbot.interactive_chat()


def run_test_questions(chatbot: RAGChatbot):
    """Run the required test questions and log results."""
    
    test_questions = [
        "What are the production 'Do's' for RAG?",
        "What is the difference between standard retrieval and the ColPali approach?",
        "Why is hybrid search better than vector-only search?"
    ]
    
    print("\n" + "=" * 70)
    print("  Running Test Questions")
    print("=" * 70)
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Question {i}] {question}")
        print("-" * 60)
        
        response = chatbot.ask(question, verbose=True)
        
        print(f"\n[Answer]\n{response.answer}")
        
        sources = list(set(chunk.source for chunk in response.sources))
        print(f"\n[Sources]: {', '.join(sources)}")
        
        results.append({
            "question": question,
            "answer": response.answer,
            "sources": sources
        })
        
        print("\n" + "=" * 70)
    
    # Save results to log file
    log_path = Path("answer_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("RAG Chatbot - Test Questions Log\n")
        f.write("=" * 70 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"Question {i}: {result['question']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Answer:\n{result['answer']}\n\n")
            f.write(f"Sources: {', '.join(result['sources'])}\n")
            f.write("\n" + "=" * 70 + "\n\n")
    
    print(f"\n✓ Results saved to: {log_path.absolute()}")


if __name__ == "__main__":
    main()
