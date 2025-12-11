# RAG Chatbot

A complete Retrieval-Augmented Generation (RAG) chatbot built that processes a knowledge base from multiple file formats (PDFs and audio/video).

## Features

- **Multi-format Ingestion**: Processes both PDF documents and audio/video files
- **Local Processing**: Uses free, local models - no API keys required
- **Persistent Vector Store**: ChromaDB for efficient similarity search
- **Semantic Chunking**: Intelligent text splitting with overlap for context preservation
- **Interactive Chat**: Ask questions about the lecture content

## Technology Stack

| Component | Tool |
|-----------|------|
| PDF Extraction | PyMuPDF (fitz) |
| Audio Transcription | OpenAI Whisper (local) |
| Text Chunking | LangChain |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| LLM | HuggingFace FLAN-T5 |

## Project Structure

```
├── Data/                      # Source files (PDFs + audio)
├── rag_chatbot/               # Main package
│   ├── __init__.py
│   ├── pdf_loader.py          # PDF text extraction
│   ├── audio_transcriber.py   # Whisper transcription
│   ├── chunker.py             # Text chunking
│   ├── embedder.py            # Embedding generation
│   ├── vector_store.py        # ChromaDB operations
│   ├── retriever.py           # Similarity search
│   ├── generator.py           # LLM response generation
│   └── chatbot.py             # Main RAG pipeline
├── main.py                    # Entry point
├── test_chatbot.py            # Test script
├── requirements.txt           # Dependencies
├── answer_log.txt             # Generated test answers
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Misrilal-Sah/rag-chatbot-genai.git
cd rag-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg (required for audio processing):
   - Windows: Download from https://ffmpeg.org/download.html and add to PATH
   - Or use: `winget install FFmpeg`

5. **Add Video Files to Data Folder**:
   
   The PDF slides are included in this repo. Download the lecture videos and place them in the `Data/` folder:
   - `1 part. RAG Intro.mp4` - RAG Introduction lecture
   - `2 part Databases for GenAI.mp4` - Databases for GenAI lecture
   
   > **Note**: Video files are too large for GitHub (>100MB). Download from your AI Academy Google Drive.

## Usage

### First Run (Index Documents)

Place your PDF and audio/video files in the `Data/` folder, then run:

```bash
python main.py
```

This will:
1. Extract text from PDFs
2. Transcribe audio/video (may take 10-20 minutes on first run)
3. Chunk and embed all text
4. Store in ChromaDB
5. Start interactive chat

### Run Test Questions

```bash
python test_chatbot.py
```

This runs the 3 required test questions and saves results to `answer_log.txt`.

### Interactive Chat

```bash
python main.py
```

Then ask questions like:
- "What are the production 'Do's' for RAG?"
- "What is hybrid search?"
- "Explain vector databases"

Type `quit` to exit.

### Force Reindex

If you add new files, reindex with:
```bash
python main.py --reindex
```

## Test Questions

The chatbot was tested with these questions:

1. "What are the production 'Do's' for RAG?"
2. "What is the difference between standard retrieval and the ColPali approach?"
3. "Why is hybrid search better than vector-only search?"

See `answer_log.txt` for the generated answers.

## Notes

- **First-time transcription** of the video file takes 10-20 minutes. Results are cached for subsequent runs.
- Uses the **lighter FLAN-T5-base** model for faster inference. For better quality, modify `chatbot.py` to use `use_light_llm=False`.
- **Whisper model size** can be changed with `--whisper-model small/medium/large` for better transcription accuracy (slower).

## Author

Built as part of the AI Academy GenAI course assignment.
