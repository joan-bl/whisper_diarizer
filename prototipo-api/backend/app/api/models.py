from pydantic import BaseModel, Field
from typing import Optional, List


class TranscriptionRequest(BaseModel):
    language: str = Field("English", description="Language of the audio")
    model: str = Field("base", description="Whisper model size to use")


class TranscriptionResponse(BaseModel):
    transcript: str = Field(..., description="Transcribed text")
    file_name: str = Field(..., description="Original filename")


class DiarizationResponse(BaseModel):
    transcript: str = Field(..., description="Raw transcribed text")
    diarized_transcript: str = Field(..., description="Transcript with speaker labels")
    speakers: List[str] = Field(..., description="List of identified speakers")
    file_name: str = Field(..., description="Original filename")
