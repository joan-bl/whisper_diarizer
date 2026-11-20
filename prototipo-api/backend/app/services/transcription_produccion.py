import os
import logging
import tempfile
from pathlib import Path
import whisper
import numpy as np
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Cache for loaded models to avoid reloading
_model_cache = {}

def get_whisper_model(model_name: str = "base"):
    """
    Load and cache a Whisper model
    
    Args:
        model_name: Name of the Whisper model to load
        
    Returns:
        Loaded whisper model
    """
    global _model_cache
    
    if model_name not in _model_cache:
        logger.info(f"Loading Whisper model: {model_name}")
        _model_cache[model_name] = whisper.load_model(model_name)
    
    return _model_cache[model_name]

def transcribe_audio(file_path: str, language: str = "English", model: str = "base") -> str:
    """
    Transcribe audio using OpenAI's Whisper model
    
    Args:
        file_path: Path to the audio file
        language: Language of the audio (or "Detect Automatically")
        model: Whisper model size to use
    
    Returns:
        Transcribed text
    """
    logger.info(f"Transcribing {Path(file_path).name} with {model} model in {language}")
    
    # Map language input to Whisper format
    whisper_language = None
    if language.lower() != "detect automatically":
        # Convert common language names to Whisper language codes
        language_map = {
            "english": "en",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "italian": "it",
            "portuguese": "pt",
            "chinese": "zh",
            "japanese": "ja",
            # Add more mappings as needed
        }
        whisper_language = language_map.get(language.lower(), language.lower())
    
    try:
        # Load the model (using cache)
        whisper_model = get_whisper_model(model)
        
        # Set transcription options
        options = {
            "verbose": False,
            "fp16": False,  # Set to True if using GPU
        }
        
        # Set language if specified
        if whisper_language:
            options["language"] = whisper_language
        
        # Transcribe the audio
        result = whisper_model.transcribe(file_path, **options)
        
        # Format the output
        transcript_text = result["text"].strip()
        
        # Add timestamps if available
        if "segments" in result:
            timestamp_transcript = "\n\n=== Detailed Transcript with Timestamps ===\n\n"
            for segment in result["segments"]:
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                text = segment["text"].strip()
                timestamp_transcript += f"[{start_time} --> {end_time}] {text}\n"
            
            transcript_text += timestamp_transcript
        
        return transcript_text
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise RuntimeError(f"Transcription failed: {str(e)}")

def format_timestamp(seconds: float) -> str:
    """
    Format seconds into HH:MM:SS.mmm
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
