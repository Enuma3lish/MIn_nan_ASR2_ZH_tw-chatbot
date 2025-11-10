import torch
import librosa
import numpy as np
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    pipeline  # <-- 關鍵
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
        # 初始化設備
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_device_id = 0 if self.device == "cuda" else -1 # Pipeline 需要 0 或 -1
        logger.info(f"ASR Service initializing on device: {self.device} (pipeline device: {self.torch_device_id})")

        # Initialize models as None - lazy loading
        self.chinese_pipeline = None # <-- 我們現在使用 pipeline
        self.min_nan_processor = None
        self.min_nan_model = None

    def _load_chinese_models(self):
        """Helper to load only Chinese ASR models using pipeline"""
        if self.chinese_pipeline is not None:
            return

        logger.info(f"Loading Chinese ASR pipeline: {settings.CHINESE_ASR_MODEL}")
        
        # 使用 pipeline，它會自動處理 chunking (長音訊)
        self.chinese_pipeline = pipeline(
            "automatic-speech-recognition",
            model=settings.CHINESE_ASR_MODEL,
            #
            # *** 錯誤的 'cache_dir' 參數已被移除 ***
            #
            device=self.torch_device_id # 傳入 0 (GPU) 或 -1 (CPU)
        )
        
        logger.info("Chinese ASR pipeline loaded successfully")
        
    def _load_min_nan_models(self):
        """Helper to load only Min Nan ASR models"""
        if self.min_nan_model is not None:
            return

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
        logger.info("Min Nan ASR models loaded successfully")

    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file (僅供 Min Nan 模型使用)
        """
        try:
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
        Transcribe Chinese audio to text using Whisper pipeline
        """
        start_time = time.time()

        try:
            # 在需要時才載入 pipeline
            self._load_chinese_models()

            # 準備 generate 參數
            generate_kwargs = {
                "language": "zh",
                "task": "transcribe"
            }

            # 直接將 *檔案路徑* 傳給 pipeline
            result = self.chinese_pipeline(
                audio_path,
                chunk_length_s=30,      # 強制 30 秒切割
                stride_length_s=5,      # 5 秒重疊，避免結尾被切斷
                generate_kwargs=generate_kwargs
            )
            
            transcription = result["text"].strip() # 獲取最終拼接的文字

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
        Transcribe Min Nan audio to text using Wav2Vec2 (此模型不支援長音訊)
        """
        start_time = time.time()

        try:
            # 在需要時才載入
            self._load_min_nan_models()

            # (注意: Min Nan 模型仍然有 30 秒限制，因為這段程式碼沒有實作 chunking)
            audio, sample_rate = self.preprocess_audio(audio_path)

            inputs = self.min_nan_processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )

            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                logits = self.min_nan_model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
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
