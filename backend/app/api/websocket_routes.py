from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import json
import logging
import uuid
from typing import Dict, Any

from ..models.schemas import LanguageType
from ..services.streaming_service import streaming_session_manager
from ..services.tts_service import tts_service
from ..services.translation_service import translation_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/asr")
async def websocket_asr_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time ASR (Automatic Speech Recognition)

    The client sends audio chunks and receives transcription results in real-time.

    Message format from client:
    {
        "type": "config",
        "language": "chinese" | "min_nan",
        "interim_results": true | false
    }
    OR
    {
        "type": "audio",
        "data": <base64 encoded audio bytes>
    }
    OR
    {
        "type": "stop"
    }

    Message format to client:
    {
        "type": "transcription",
        "text": "transcribed text",
        "is_final": true | false
    }
    OR
    {
        "type": "error",
        "message": "error message"
    }
    """
    await websocket.accept()

    # Create session
    session_id = str(uuid.uuid4())
    session = streaming_session_manager.create_session(session_id)

    # Session configuration
    config = {
        "language": LanguageType.CHINESE,
        "interim_results": False
    }

    logger.info(f"WebSocket ASR connection established: session={session_id}")

    try:
        while True:
            # Receive message
            data = await websocket.receive()

            if "text" in data:
                # Handle JSON messages
                try:
                    message = json.loads(data["text"])
                    message_type = message.get("type")

                    if message_type == "config":
                        # Update configuration
                        lang = message.get("language", "chinese")
                        config["language"] = LanguageType(lang)
                        config["interim_results"] = message.get("interim_results", False)

                        await websocket.send_json({
                            "type": "config_updated",
                            "language": lang,
                            "interim_results": config["interim_results"]
                        })

                    elif message_type == "stop":
                        # Get final transcription
                        final_text = await session.transcribe_final(config["language"])

                        if final_text:
                            await websocket.send_json({
                                "type": "transcription",
                                "text": final_text,
                                "is_final": True
                            })

                        # Reset buffer for next session
                        session.reset_buffer()

                        await websocket.send_json({
                            "type": "stopped"
                        })

                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Unknown message type: {message_type}"
                        })

                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON"
                    })

            elif "bytes" in data:
                # Handle audio bytes
                try:
                    audio_chunk = data["bytes"]

                    # Transcribe stream
                    text = await session.transcribe_stream(
                        audio_chunk,
                        config["language"],
                        interim_results=config["interim_results"]
                    )

                    if text:
                        await websocket.send_json({
                            "type": "transcription",
                            "text": text,
                            "is_final": not config["interim_results"]
                        })

                except Exception as e:
                    logger.error(f"Error processing audio: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })

    except WebSocketDisconnect:
        logger.info(f"WebSocket ASR disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"WebSocket ASR error: {str(e)}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    finally:
        # Cleanup session
        streaming_session_manager.remove_session(session_id)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()


@router.websocket("/ws/voice-chat")
async def websocket_voice_chat_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice chat with translation

    The client sends audio in one language and receives audio in another language.

    Message format from client:
    {
        "type": "config",
        "source_language": "chinese" | "min_nan",
        "target_language": "chinese" | "min_nan"
    }
    OR
    {
        "type": "audio",
        "data": <base64 encoded audio bytes>
    }
    OR
    {
        "type": "stop"
    }

    Message format to client:
    {
        "type": "transcription",
        "text": "transcribed text"
    }
    OR
    {
        "type": "translation",
        "text": "translated text"
    }
    OR
    {
        "type": "audio_ready",
        "audio_url": "/api/v1/audio/filename.wav"
    }
    OR
    {
        "type": "error",
        "message": "error message"
    }
    """
    await websocket.accept()

    # Create session
    session_id = str(uuid.uuid4())
    session = streaming_session_manager.create_session(session_id)

    # Session configuration
    config = {
        "source_language": LanguageType.CHINESE,
        "target_language": LanguageType.MIN_NAN
    }

    logger.info(f"WebSocket Voice Chat connection established: session={session_id}")

    try:
        while True:
            data = await websocket.receive()

            if "text" in data:
                try:
                    message = json.loads(data["text"])
                    message_type = message.get("type")

                    if message_type == "config":
                        # Update configuration
                        src_lang = message.get("source_language", "chinese")
                        tgt_lang = message.get("target_language", "min_nan")

                        config["source_language"] = LanguageType(src_lang)
                        config["target_language"] = LanguageType(tgt_lang)

                        await websocket.send_json({
                            "type": "config_updated",
                            "source_language": src_lang,
                            "target_language": tgt_lang
                        })

                    elif message_type == "stop":
                        # Process final audio
                        # 1. Transcribe
                        transcribed_text = await session.transcribe_final(
                            config["source_language"]
                        )

                        if transcribed_text:
                            await websocket.send_json({
                                "type": "transcription",
                                "text": transcribed_text
                            })

                            # 2. Translate
                            translated_text, translation_time = translation_service.translate(
                                transcribed_text,
                                config["source_language"],
                                config["target_language"]
                            )

                            await websocket.send_json({
                                "type": "translation",
                                "text": translated_text
                            })

                            # 3. Generate speech
                            output_path, tts_time = tts_service.text_to_speech(
                                translated_text,
                                f"voice_chat_{session_id}.wav"
                            )

                            import os
                            filename = os.path.basename(output_path)

                            await websocket.send_json({
                                "type": "audio_ready",
                                "audio_url": f"/api/v1/audio/{filename}",
                                "text": translated_text
                            })

                        # Reset buffer
                        session.reset_buffer()

                        await websocket.send_json({
                            "type": "stopped"
                        })

                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON"
                    })

            elif "bytes" in data:
                # Just buffer audio for now
                try:
                    audio_chunk = data["bytes"]
                    session.add_audio_chunk(audio_chunk)

                    await websocket.send_json({
                        "type": "audio_received",
                        "size": len(audio_chunk)
                    })

                except Exception as e:
                    logger.error(f"Error processing audio: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })

    except WebSocketDisconnect:
        logger.info(f"WebSocket Voice Chat disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"WebSocket Voice Chat error: {str(e)}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    finally:
        # Cleanup session
        streaming_session_manager.remove_session(session_id)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
