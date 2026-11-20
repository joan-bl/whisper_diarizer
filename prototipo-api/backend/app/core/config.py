from pydantic_settings import BaseSettings
import os
from pathlib import Path


class Settings(BaseSettings):
    # API configuration
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG_MODE: bool = True
    
    # Upload configuration
    UPLOAD_DIR: str = "uploads"
    
    # Whisper configuration
    DEFAULT_LANGUAGE: str = "English"
    DEFAULT_MODEL: str = "base"
    
    # Create upload directory if it doesn't exist
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
