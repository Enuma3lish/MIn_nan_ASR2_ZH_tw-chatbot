import torch
import librosa
import numpy as np
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
import time
import logging
from typing import Tuple, Optional
from ..config import settings
from ..models.schemas import LanguageType

logger = logging.getLogger(__name__)


class ASRService:
    """Service for Automatic Speech Recognition (ASR) for Chinese and Min Nan"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ASR Service initializing on device: {self.device}")

        # Initialize models as None - lazy loading
        self.chinese_processor = None
        self.chinese_model = None
        self.min_nan_processor = None
        self.min_nan_model = None

        self.models_loaded = False

    def load_models(self):
        """Load ASR models (lazy loading)"""
        try:
            logger.info("Loading ASR models...")

            # Load Whisper model for Chinese ASR
            logger.info(f"Loading Chinese ASR model: {settings.CHINESE_ASR_MODEL}")
            self.chinese_processor = WhisperProcessor.from_pretrained(
                settings.CHINESE_ASR_MODEL,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            self.chinese_model = WhisperForConditionalGeneration.from_pretrained(
                settings.CHINESE_ASR_MODEL,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            self.chinese_model.to(self.device)
            self.chinese_model.eval()

            # Load Wav2Vec2 model for Min Nan ASR
            # Note: This uses a base model. For better Min Nan support, you should fine-tune this model
            logger.info(f"Loading Min Nan ASR model: {settings.MIN_NAN_ASR_MODEL}")
            self.min_nan_processor = Wav2Vec2Processor.from_pretrained(
                settings.MIN_NAN_ASR_MODEL,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            self.min_nan_model = Wav2Vec2ForCTC.from_pretrained(
                settings.MIN_NAN_ASR_MODEL,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            self.min_nan_model.to(self.device)
            self.min_nan_model.eval()

            self.models_loaded = True
            logger.info("ASR models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading ASR models: {str(e)}")
            raise

    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Load audio file and resample to target sample rate
            audio, sample_rate = librosa.load(
                audio_path,
                sr=settings.SAMPLE_RATE,
                mono=True
            )
            return audio, sample_rate
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            raise

    def transcribe_chinese(
        self,
        audio_path: str
    ) -> Tuple[str, Optional[float], float]:
        """
        Transcribe Chinese audio to text using Whisper

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (transcribed_text, confidence, processing_time)
        """
        start_time = time.time()

        try:
            if not self.models_loaded:
                self.load_models()

            # Preprocess audio
            audio, sample_rate = self.preprocess_audio(audio_path)

            # Process with Whisper
            input_features = self.chinese_processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features

            input_features = input_features.to(self.device)

            # Generate transcription
            with torch.no_grad():
                # Force Chinese language
                forced_decoder_ids = self.chinese_processor.get_decoder_prompt_ids(
                    language="zh",
                    task="transcribe"
                )
                predicted_ids = self.chinese_model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids
                )

            # Decode transcription
            transcription = self.chinese_processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            processing_time = time.time() - start_time

            logger.info(f"Chinese transcription completed in {processing_time:.2f}s")
            return transcription, None, processing_time

        except Exception as e:
            logger.error(f"Error in Chinese transcription: {str(e)}")
            raise

    def transcribe_min_nan(
        self,
        audio_path: str
    ) -> Tuple[str, Optional[float], float]:
        """
        Transcribe Min Nan audio to text using Wav2Vec2

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (transcribed_text, confidence, processing_time)
        """
        start_time = time.time()

        try:
            if not self.models_loaded:
                self.load_models()

            # Preprocess audio
            audio, sample_rate = self.preprocess_audio(audio_path)

            # Process with Wav2Vec2
            inputs = self.min_nan_processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )

            input_values = inputs.input_values.to(self.device)

            # Generate transcription
            with torch.no_grad():
                logits = self.min_nan_model(input_values).logits

            # Get predicted ids
            predicted_ids = torch.argmax(logits, dim=-1)

            # Decode transcription
            transcription = self.min_nan_processor.batch_decode(predicted_ids)[0]

            processing_time = time.time() - start_time

            logger.info(f"Min Nan transcription completed in {processing_time:.2f}s")
            return transcription, None, processing_time

        except Exception as e:
            logger.error(f"Error in Min Nan transcription: {str(e)}")
            raise

    def transcribe(
        self,
        audio_path: str,
        language: LanguageType
    ) -> Tuple[str, Optional[float], float]:
        """
        Transcribe audio to text based on language

        Args:
            audio_path: Path to audio file
            language: Language type (CHINESE or MIN_NAN)

        Returns:
            Tuple of (transcribed_text, confidence, processing_time)
        """
        logger.info(f"Transcribing audio with language: {language}")

        if language == LanguageType.CHINESE or language == LanguageType.ZH_TW:
            return self.transcribe_chinese(audio_path)
        elif language == LanguageType.MIN_NAN:
            return self.transcribe_min_nan(audio_path)
        else:
            raise ValueError(f"Unsupported language: {language}")


# Global ASR service instance
asr_service = ASRService()
