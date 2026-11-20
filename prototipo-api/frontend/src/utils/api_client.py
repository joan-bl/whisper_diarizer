import requests
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class APIClient:
    """Client for interacting with the Whisper API"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000/api/v1"):
        self.base_url = base_url
        
    def check_health(self) -> Dict[str, Any]:
        """Check if the API is running"""
        try:
            response = requests.get(f"{self.base_url}/")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API health check failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def transcribe_audio(self, file_path: str, language: str, model: str) -> Optional[str]:
        """
        Send audio file to API for transcription
        
        Args:
            file_path: Path to the audio file
            language: Language of the audio
            model: Whisper model size to use
            
        Returns:
            Transcribed text or None if request failed
        """
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                params = {"language": language, "model": model}
                response = requests.post(
                    f"{self.base_url}/transcribe", 
                    files=files, 
                    params=params,
                    timeout=300  # Extended timeout for large files
                )
            
            response.raise_for_status()
            return response.json().get("transcript")
        except requests.RequestException as e:
            logger.error(f"Transcription request failed: {e}")
            return None
