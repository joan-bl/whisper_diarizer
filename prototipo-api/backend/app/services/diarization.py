import os
import logging
import numpy as np
from pathlib import Path
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment
from typing import List, Dict, Any, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the diarization pipeline (cached)
_diarization_pipeline = None

def get_diarization_pipeline():
    """
    Load and cache the diarization pipeline
    
    Returns:
        Loaded pyannote.audio pipeline
    """
    global _diarization_pipeline
    
    if _diarization_pipeline is None:
        # Check if HF_TOKEN environment variable is set
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN environment variable not set. Diarization may fail.")
            
        try:
            logger.info("Loading speaker diarization pipeline")
            _diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.0", 
                use_auth_token=hf_token
            )
            
            # Set pipeline to run on CPU if no GPU is available
            if not torch.cuda.is_available():
                logger.info("No GPU detected, running diarization on CPU")
                _diarization_pipeline.to(torch.device("cpu"))
            
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            raise RuntimeError(f"Failed to initialize diarization: {str(e)}")
    
    return _diarization_pipeline

def diarize_audio(file_path: str) -> List[Dict[str, Any]]:
    """
    Perform speaker diarization on an audio file
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        List of speaker segments with start/end times and speaker labels
    """
    try:
        logger.info(f"Diarizing speakers in {Path(file_path).name}")
        
        # Get the diarization pipeline
        pipeline = get_diarization_pipeline()
        
        # Run diarization
        diarization = pipeline(file_path)
        
        # Convert to a format we can use
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "duration": turn.end - turn.start
            })
        
        return segments
        
    except Exception as e:
        logger.error(f"Error during diarization: {str(e)}")
        raise RuntimeError(f"Diarization failed: {str(e)}")

def combine_transcript_with_diarization(
    transcript_segments: List[Dict[str, Any]], 
    diarization_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Combine whisper transcript with speaker diarization
    
    Args:
        transcript_segments: Segments from Whisper transcript
        diarization_segments: Segments from speaker diarization
        
    Returns:
        Combined segments with text and speaker information
    """
    result = []
    
    for trans_seg in transcript_segments:
        trans_start = trans_seg["start"]
        trans_end = trans_seg["end"]
        trans_text = trans_seg["text"]
        
        # Find overlapping diarization segments
        overlapping = []
        for diar_seg in diarization_segments:
            diar_start = diar_seg["start"]
            diar_end = diar_seg["end"]
            
            # Check for overlap
            if max(trans_start, diar_start) < min(trans_end, diar_end):
                overlapping.append(diar_seg)
        
        # If we found overlapping segments, assign the speaker with most overlap
        if overlapping:
            best_speaker = find_best_speaker(trans_start, trans_end, overlapping)
            result.append({
                "start": trans_start,
                "end": trans_end,
                "text": trans_text,
                "speaker": best_speaker
            })
        else:
            # No overlap found, just use the transcript segment
            result.append({
                "start": trans_start,
                "end": trans_end,
                "text": trans_text,
                "speaker": "UNKNOWN"
            })
    
    return result

def find_best_speaker(
    segment_start: float, 
    segment_end: float, 
    speaker_segments: List[Dict[str, Any]]
) -> str:
    """
    Find which speaker has the most overlap with a segment
    
    Args:
        segment_start: Start time of segment
        segment_end: End time of segment
        speaker_segments: List of speaker segments to check
        
    Returns:
        Label of the speaker with most overlap
    """
    speaker_overlap = {}
    
    for spk_seg in speaker_segments:
        speaker = spk_seg["speaker"]
        overlap_start = max(segment_start, spk_seg["start"])
        overlap_end = min(segment_end, spk_seg["end"])
        overlap = max(0, overlap_end - overlap_start)
        
        if speaker not in speaker_overlap:
            speaker_overlap[speaker] = 0
        speaker_overlap[speaker] += overlap
    
    # Find speaker with most overlap
    best_speaker = max(speaker_overlap.items(), key=lambda x: x[1])[0]
    return best_speaker

def format_diarized_transcript(combined_segments: List[Dict[str, Any]]) -> str:
    """
    Format diarized transcript as readable text
    
    Args:
        combined_segments: Combined transcript and diarization data
        
    Returns:
        Formatted transcript with speaker labels
    """
    result = []
    current_speaker = None
    
    for segment in combined_segments:
        speaker = segment["speaker"]
        text = segment["text"].strip()
        start_time = format_timestamp(segment["start"])
        
        # Only show speaker change or initial speaker
        if speaker != current_speaker:
            result.append(f"\n[{start_time}] {speaker}: {text}")
            current_speaker = speaker
        else:
            # Continue previous speaker's text
            result.append(f" {text}")
    
    return "".join(result)

def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
