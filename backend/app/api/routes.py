from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
import uuid
import logging
import aiofiles
from typing import Optional

from ..models.schemas import (
    ASRResponse,
    TTSRequest,
    TTSResponse,
    VoiceConversionResponse,
    HealthResponse,
    LanguageType,
)
from ..services.asr_service import asr_service
from ..services.tts_service import tts_service
from ..services.translation_service import translation_service
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


def cleanup_file(file_path: str):
    """Background task to cleanup uploaded/generated files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns API status and model loading status
    """
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        models_loaded=(
            asr_service.models_loaded and
            tts_service.models_loaded
            # Translation service can work without neural models (dictionary fallback)
        )
    )


@router.post("/asr/transcribe", response_model=ASRResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    language: LanguageType = Form(..., description="Language of the audio (chinese or min_nan)")
):
    """
    Transcribe audio to text (ASR - Automatic Speech Recognition)

    - **audio_file**: Audio file in supported formats (wav, mp3, m4a, ogg, flac)
    - **language**: Language of the audio (chinese, min_nan, or zh_tw)

    Returns transcribed text with metadata
    """
    try:
        # Validate file extension
        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        if file_ext not in settings.ALLOWED_AUDIO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed formats: {settings.ALLOWED_AUDIO_FORMATS}"
            )

        # Save uploaded file
        file_id = uuid.uuid4().hex
        file_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{file_ext}")

        async with aiofiles.open(file_path, 'wb') as f:
            content = await audio_file.read()
            await f.write(content)

        logger.info(f"Saved uploaded file to: {file_path}")

        # Transcribe audio
        text, confidence, processing_time = asr_service.transcribe(
            file_path,
            language
        )

        # Schedule cleanup of uploaded file
        background_tasks.add_task(cleanup_file, file_path)

        return ASRResponse(
            text=text,
            language=language,
            confidence=confidence,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")
        # Cleanup on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tts/synthesize", response_model=TTSResponse)
async def synthesize_speech(
    request: TTSRequest,
    background_tasks: BackgroundTasks
):
    """
    Convert text to speech (TTS - Text-to-Speech)

    - **text**: Text to convert to speech
    - **source_language**: Source language of the text (default: chinese)
    - **target_language**: Target language for speech output (default: min_nan)

    Returns URL to download the generated audio file
    """
    try:
        logger.info(f"TTS request: {request.text[:50]}...")

        # Translate and convert to speech
        translated_text, output_path, processing_time = tts_service.translate_and_speak(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language
        )

        # Generate URL for audio file
        filename = os.path.basename(output_path)
        audio_url = f"/api/v1/audio/{filename}"

        return TTSResponse(
            audio_url=audio_url,
            text=translated_text,
            source_language=request.source_language,
            target_language=request.target_language,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error in synthesize_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice-conversion", response_model=VoiceConversionResponse)
async def voice_to_voice_conversion(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file to convert"),
    source_language: LanguageType = Form(
        default=LanguageType.CHINESE,
        description="Source language of the audio"
    ),
    target_language: LanguageType = Form(
        default=LanguageType.MIN_NAN,
        description="Target language for output audio"
    )
):
    """
    Convert voice from one language to another (Voice-to-Voice)

    This endpoint:
    1. Transcribes the input audio to text (ASR)
    2. Translates the text if needed
    3. Converts the translated text to speech in target language (TTS)

    - **audio_file**: Audio file in supported formats
    - **source_language**: Language of the input audio (default: chinese)
    - **target_language**: Language for output audio (default: min_nan)

    Returns transcribed text and URL to download the converted audio
    """
    try:
        # Validate file extension
        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        if file_ext not in settings.ALLOWED_AUDIO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed formats: {settings.ALLOWED_AUDIO_FORMATS}"
            )

        # Save uploaded file
        file_id = uuid.uuid4().hex
        input_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{file_ext}")

        async with aiofiles.open(input_path, 'wb') as f:
            content = await audio_file.read()
            await f.write(content)

        logger.info(f"Processing voice conversion: {source_language} -> {target_language}")

        # Step 1: Transcribe input audio
        transcribed_text, _, asr_time = asr_service.transcribe(
            input_path,
            source_language
        )

        # Step 2: Translate and convert to speech
        translated_text, output_path, tts_time = tts_service.translate_and_speak(
            text=transcribed_text,
            source_language=source_language,
            target_language=target_language
        )

        # Generate URL for output audio
        filename = os.path.basename(output_path)
        audio_url = f"/api/v1/audio/{filename}"

        total_time = asr_time + tts_time

        # Schedule cleanup of uploaded file
        background_tasks.add_task(cleanup_file, input_path)

        return VoiceConversionResponse(
            transcribed_text=transcribed_text,
            audio_url=audio_url,
            source_language=source_language,
            target_language=target_language,
            processing_time=total_time
        )

    except Exception as e:
        logger.error(f"Error in voice_to_voice_conversion: {str(e)}")
        # Cleanup on error
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """
    Download generated audio file

    - **filename**: Name of the audio file to download
    """
    file_path = os.path.join(settings.OUTPUT_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=filename
    )


@router.post("/models/load")
async def load_models():
    """
    Manually trigger loading of AI models

    This can be useful for preloading models on server startup
    """
    try:
        if not asr_service.models_loaded:
            asr_service.load_models()

        if not tts_service.models_loaded:
            tts_service.load_models()

        if not translation_service.models_loaded:
            translation_service.load_models()

        return {
            "status": "success",
            "message": "Models loaded successfully",
            "asr_loaded": asr_service.models_loaded,
            "tts_loaded": tts_service.models_loaded,
            "translation_loaded": translation_service.models_loaded
        }
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
