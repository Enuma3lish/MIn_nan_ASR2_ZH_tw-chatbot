from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings and configuration"""

    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Min Nan & Chinese Voice Chatbot API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API for Chinese and Min Nan voice-to-text and text-to-voice conversion"

    # CORS Settings
    BACKEND_CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8000",
        "http://localhost:8080",
    ]

    # Model Settings
    CHINESE_ASR_MODEL: str = "openai/whisper-large-v3"  # For Chinese ASR
    MIN_NAN_ASR_MODEL: str = "facebook/wav2vec2-large-xlsr-53"  # Can be fine-tuned for Min Nan
    MIN_NAN_TTS_MODEL: str = "facebook/mms-tts-nan"  # Min Nan TTS model

    # File Upload Settings
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_AUDIO_FORMATS: list = [".wav", ".mp3", ".m4a", ".ogg", ".flac"]

    # Audio Processing Settings
    SAMPLE_RATE: int = 16000  # Standard sample rate for ASR models

    # Device Settings (CPU/CUDA)
    DEVICE: str = "cpu"  # Will auto-detect CUDA if available

    # Cache Settings
    MODEL_CACHE_DIR: str = "./model_cache"

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()

# Create necessary directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
