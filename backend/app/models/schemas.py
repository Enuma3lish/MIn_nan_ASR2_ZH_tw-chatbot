from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class LanguageType(str, Enum):
    """Supported language types"""
    CHINESE = "chinese"
    MIN_NAN = "min_nan"
    ZH_TW = "zh_tw"


class ASRRequest(BaseModel):
    """Request model for Automatic Speech Recognition"""
    language: LanguageType = Field(
        ...,
        description="Language of the input audio (chinese or min_nan)"
    )


class ASRResponse(BaseModel):
    """Response model for ASR"""
    text: str = Field(..., description="Transcribed text from audio")
    language: LanguageType = Field(..., description="Detected/specified language")
    confidence: Optional[float] = Field(None, description="Confidence score if available")
    processing_time: float = Field(..., description="Processing time in seconds")


class TTSRequest(BaseModel):
    """Request model for Text-to-Speech"""
    text: str = Field(..., description="Text to convert to speech")
    source_language: LanguageType = Field(
        default=LanguageType.CHINESE,
        description="Source language of the text"
    )
    target_language: LanguageType = Field(
        default=LanguageType.MIN_NAN,
        description="Target language for speech output"
    )


class TTSResponse(BaseModel):
    """Response model for TTS"""
    audio_url: str = Field(..., description="URL to download the generated audio")
    text: str = Field(..., description="Original text")
    source_language: LanguageType = Field(..., description="Source language")
    target_language: LanguageType = Field(..., description="Target language")
    processing_time: float = Field(..., description="Processing time in seconds")


class VoiceConversionRequest(BaseModel):
    """Request model for voice-to-voice conversion (Chinese voice -> Min Nan voice)"""
    source_language: LanguageType = Field(
        default=LanguageType.CHINESE,
        description="Source language of the input audio"
    )
    target_language: LanguageType = Field(
        default=LanguageType.MIN_NAN,
        description="Target language for output audio"
    )


class VoiceConversionResponse(BaseModel):
    """Response model for voice conversion"""
    transcribed_text: str = Field(..., description="Transcribed text from input audio")
    audio_url: str = Field(..., description="URL to download the converted audio")
    source_language: LanguageType = Field(..., description="Source language")
    target_language: LanguageType = Field(..., description="Target language")
    processing_time: float = Field(..., description="Total processing time in seconds")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(..., description="Whether models are loaded")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
