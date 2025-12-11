"""
Audio Transcription Module
Uses OpenAI's Whisper (local) for speech-to-text transcription.
"""

import os
import json
from pathlib import Path
from typing import Optional
import hashlib


def get_file_hash(file_path: str) -> str:
    """Get MD5 hash of file for caching purposes."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[:12]


def transcribe_audio(
    audio_path: str,
    model_name: str = "base",
    cache_dir: Optional[str] = None,
    force_retranscribe: bool = False
) -> str:
    """
    Transcribe audio/video file using local Whisper model.
    
    Args:
        audio_path: Path to audio or video file
        model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        cache_dir: Directory to cache transcriptions (default: transcripts/)
        force_retranscribe: If True, ignore cached transcription
        
    Returns:
        Transcribed text
    """
    import whisper
    
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Set up cache directory
    if cache_dir is None:
        cache_dir = path.parent.parent / "transcripts"
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Create cache file path
    file_hash = get_file_hash(audio_path)
    cache_file = cache_path / f"{path.stem}_{file_hash}.txt"
    cache_meta = cache_path / f"{path.stem}_{file_hash}_meta.json"
    
    # Check for cached transcription
    if not force_retranscribe and cache_file.exists():
        print(f"✓ Loading cached transcription from: {cache_file.name}")
        return cache_file.read_text(encoding="utf-8")
    
    print(f"Transcribing: {path.name}")
    print(f"Model: {model_name} (this may take several minutes for large files...)")
    
    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model(model_name)
    
    # Transcribe
    print("Starting transcription...")
    result = model.transcribe(
        str(audio_path),
        verbose=True,
        language="en"
    )
    
    transcription = result["text"]
    
    # Cache the transcription
    cache_file.write_text(transcription, encoding="utf-8")
    
    # Save metadata
    meta = {
        "source_file": path.name,
        "model": model_name,
        "language": result.get("language", "en"),
        "duration_seconds": result.get("duration", 0)
    }
    cache_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
    print(f"✓ Transcription complete! Saved to: {cache_file.name}")
    print(f"  Length: {len(transcription)} characters")
    
    return transcription


def transcribe_multiple_files(
    file_paths: list,
    model_name: str = "base",
    cache_dir: Optional[str] = None
) -> dict:
    """
    Transcribe multiple audio/video files.
    
    Args:
        file_paths: List of paths to audio/video files
        model_name: Whisper model size
        cache_dir: Directory to cache transcriptions
        
    Returns:
        Dictionary mapping file names to transcriptions
    """
    results = {}
    
    for file_path in file_paths:
        path = Path(file_path)
        try:
            text = transcribe_audio(file_path, model_name, cache_dir)
            results[path.name] = text
        except Exception as e:
            print(f"✗ Error transcribing {path.name}: {e}")
            results[path.name] = ""
    
    return results


def load_all_audio_from_directory(
    directory: str,
    extensions: tuple = (".mp4", ".mp3", ".wav", ".m4a", ".webm"),
    model_name: str = "base"
) -> dict:
    """
    Load and transcribe all audio/video files in a directory.
    
    Args:
        directory: Path to directory
        extensions: Tuple of valid file extensions
        model_name: Whisper model size
        
    Returns:
        Dictionary mapping file names to transcriptions
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise NotADirectoryError(f"Directory not found: {directory}")
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(dir_path.glob(f"*{ext}"))
    
    if not audio_files:
        print(f"No audio/video files found in {directory}")
        return {}
    
    print(f"Found {len(audio_files)} audio/video file(s)")
    return transcribe_multiple_files([str(p) for p in audio_files], model_name)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        text = transcribe_audio(audio_path)
        print("\n--- First 1000 characters of transcription ---")
        print(text[:1000])
