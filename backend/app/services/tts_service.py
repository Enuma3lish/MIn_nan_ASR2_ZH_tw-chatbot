import torch
import numpy as np
from transformers import VitsModel, VitsTokenizer
import time
import logging
import os
import uuid
from scipy.io import wavfile
from typing import Tuple
from ..config import settings
from ..models.schemas import LanguageType

logger = logging.getLogger(__name__)


class TTSService:
    """Service for Text-to-Speech (TTS) using Facebook MMS-TTS for Min Nan"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"TTS Service initializing on device: {self.device}")

        # Initialize models as None - lazy loading
        self.tokenizer = None
        self.model = None
        self.models_loaded = False

    def load_models(self):
        """Load TTS models (lazy loading)"""
        try:
            logger.info("Loading TTS models...")

            # Load Facebook MMS-TTS model for Min Nan
            logger.info(f"Loading Min Nan TTS model: {settings.MIN_NAN_TTS_MODEL}")
            self.tokenizer = VitsTokenizer.from_pretrained(
                settings.MIN_NAN_TTS_MODEL,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            self.model = VitsModel.from_pretrained(
                settings.MIN_NAN_TTS_MODEL,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            self.model.to(self.device)
            self.model.eval()

            self.models_loaded = True
            logger.info("TTS models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading TTS models: {str(e)}")
            raise

    def text_to_speech(
        self,
        text: str,
        output_filename: str = None
    ) -> Tuple[str, float]:
        """
        Convert text to speech using Facebook MMS-TTS

        Args:
            text: Text to convert to speech
            output_filename: Optional custom output filename

        Returns:
            Tuple of (output_file_path, processing_time)
        """
        start_time = time.time()

        try:
            if not self.models_loaded:
                self.load_models()

            logger.info(f"Converting text to speech: {text[:50]}...")

            # Tokenize input text
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate speech
            with torch.no_grad():
                outputs = self.model(**inputs)
                waveform = outputs.waveform[0].cpu().numpy()

            # Generate output filename if not provided
            if output_filename is None:
                output_filename = f"tts_{uuid.uuid4().hex}.wav"

            output_path = os.path.join(settings.OUTPUT_DIR, output_filename)

            # Save audio file
            # MMS-TTS typically outputs at 16kHz
            sample_rate = 16000
            # Normalize waveform to int16 range
            waveform_int16 = np.int16(waveform * 32767)
            wavfile.write(output_path, sample_rate, waveform_int16)

            processing_time = time.time() - start_time

            logger.info(
                f"TTS completed in {processing_time:.2f}s. "
                f"Output saved to: {output_path}"
            )

            return output_path, processing_time

        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {str(e)}")
            raise

    def translate_and_speak(
        self,
        text: str,
        source_language: LanguageType,
        target_language: LanguageType,
        output_filename: str = None
    ) -> Tuple[str, str, float]:
        """
        Translate text (if needed) and convert to speech

        Args:
            text: Input text
            source_language: Source language of the text
            target_language: Target language for speech output
            output_filename: Optional custom output filename

        Returns:
            Tuple of (translated_text, output_file_path, processing_time)
        """
        start_time = time.time()

        try:
            # For now, we'll use the text as-is
            # In a production system, you would add translation here
            # e.g., using a Chinese to Min Nan translation model
            translated_text = text

            if source_language == LanguageType.CHINESE and target_language == LanguageType.MIN_NAN:
                logger.info("Translation from Chinese to Min Nan would happen here")
                # TODO: Implement translation model
                # For now, we'll just pass through the text
                # You might want to use a translation model or dictionary
                pass

            # Convert to speech (always Min Nan for this model)
            output_path, _ = self.text_to_speech(translated_text, output_filename)

            processing_time = time.time() - start_time

            return translated_text, output_path, processing_time

        except Exception as e:
            logger.error(f"Error in translate and speak: {str(e)}")
            raise


# Global TTS service instance
tts_service = TTSService()
