"""
Test Script for RAG Chatbot

This script runs the 3 required test questions and logs the results.
Run this after indexing to generate the answer_log.txt file.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag_chatbot.chatbot import RAGChatbot


def run_tests():
    """Run the required test questions and save results to log file."""
    
    # Test questions from the assignment
    TEST_QUESTIONS = [
        "What are the production 'Do's' for RAG?",
        "What is the difference between standard retrieval and the ColPali approach?",
        "Why is hybrid search better than vector-only search?"
    ]
    
    print("\n" + "=" * 70)
    print("  RAG Chatbot - Test Script")
    print("=" * 70)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Initialize chatbot
    chatbot = RAGChatbot(
        data_directory="./Data",
        persist_directory="./chroma_db",
        collection_name="genai_lectures",
        use_light_llm=True
    )
    
    # Check if indexed
    if not chatbot.is_indexed:
        print("\nKnowledge base is empty. Indexing documents first...")
        chatbot.index_documents()
    
    # Prepare log content
    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append("RAG CHATBOT - TEST RESULTS LOG")
    log_lines.append("=" * 70)
    log_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Knowledge Base Size: {chatbot.vector_store.count()} chunks")
    log_lines.append("=" * 70)
    log_lines.append("")
    
    # Run each test question
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'=' * 70}")
        print(f"QUESTION {i}: {question}")
        print("=" * 70)
        
        # Get response
        response = chatbot.ask(question, top_k=5, verbose=True)
        
        # Print results
        print(f"\nANSWER:")
        print("-" * 40)
        print(response.answer)
        
        sources = list(set(chunk.source for chunk in response.sources))
        print(f"\nSOURCES: {', '.join(sources)}")
        
        # Add to log
        log_lines.append(f"QUESTION {i}:")
        log_lines.append(question)
        log_lines.append("")
        log_lines.append("ANSWER:")
        log_lines.append(response.answer)
        log_lines.append("")
        log_lines.append(f"SOURCES: {', '.join(sources)}")
        log_lines.append("")
        log_lines.append("RETRIEVED CONTEXT (top 3):")
        for j, chunk in enumerate(response.sources[:3], 1):
            log_lines.append(f"  [{j}] Source: {chunk.source} (score: {chunk.relevance_score:.3f})")
            log_lines.append(f"      Text: {chunk.text[:200]}...")
        log_lines.append("")
        log_lines.append("-" * 70)
        log_lines.append("")
    
    # Save log file
    log_path = Path("answer_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    
    print(f"\n{'=' * 70}")
    print(f"✓ Test complete! Results saved to: {log_path.absolute()}")
    print("=" * 70)
    
    return log_path


if __name__ == "__main__":
    run_tests()
