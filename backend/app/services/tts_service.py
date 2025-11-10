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
from pypinyin import pinyin, Style  # <-- *** 新增 pypinyin 依賴 ***

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

    def convert_hanzi_to_pinyin(self, text: str) -> str:
        """
        Converts Chinese Hanzi text to Pinyin string, as MMS-TTS only accepts Pinyin.
        """
        try:
            # 偵測是否已經是拼音 (包含英文字母和數字)
            if all(ord(c) < 128 for c in text.replace(" ", "")):
                 logger.info("Input is already Pinyin, skipping conversion.")
                 return text

            logger.info("Input is Hanzi, converting to Pinyin...")
            # style=Style.TONE3 產生 'ni3 hao3' 樣式
            pinyin_list = pinyin(text, style=Style.TONE3, heteronym=False)
            # 將 [['ni3'], ['hao3']] 轉換為 "ni3 hao3"
            text_pinyin = " ".join([item[0] for item in pinyin_list])
            logger.info(f"Pinyin conversion result: {text_pinyin}")
            return text_pinyin
        except Exception as e:
            logger.error(f"Error during Pinyin conversion: {e}")
            return text # 轉換失敗時，回傳原文

    def text_to_speech(
        self,
        text: str,
        output_filename: str = None
    ) -> Tuple[str, float]:
        """
        Convert text to speech using Facebook MMS-TTS
        """
        start_time = time.time()

        try:
            if not self.models_loaded:
                self.load_models()

            logger.info(f"Converting text to speech: {text[:50]}...")
            
            # --- 錯誤修正 START ---
            # ** 這是修復 'input size 0' 錯誤的關鍵 **
            # 將傳入的文字 (無論是漢字還是拼音) 統一轉換為拼音
            text_pinyin = self.convert_hanzi_to_pinyin(text)
            # --- 錯誤修正 END ---

            # Tokenize input text (使用 Pinyin)
            inputs = self.tokenizer(text_pinyin, return_tensors="pt")
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # VITS 模型的嵌入層 (indices) 期望 LongTensor (整數)
            if 'input_ids' in inputs:
                inputs['input_ids'] = inputs['input_ids'].long()

            # Generate speech
            with torch.no_grad():
                outputs = self.model(**inputs)
                waveform = outputs.waveform[0].cpu().numpy()

            if output_filename is None:
                output_filename = f"tts_{uuid.uuid4().hex}.wav"

            output_path = os.path.join(settings.OUTPUT_DIR, output_filename)
            
            sample_rate = 16000
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
        output_filename: str = None,
        use_neural_translation: bool = True
    ) -> Tuple[str, str, float]:
        """
        Translate text (if needed) and convert to speech
        """
        start_time = time.time()

        try:
            from .translation_service import translation_service

            if source_language != target_language:
                logger.info(f"Translating from {source_language} to {target_language}")
                translated_text, translation_time = translation_service.translate(
                    text,
                    source_language,
                    target_language,
                    use_neural=use_neural_translation
                )
                logger.info(
                    f"Translation completed in {translation_time:.2f}s: "
                    f"'{text}' -> '{translated_text}'"
                )
            else:
                translated_text = text

            # 傳遞給 TTS (text_to_speech 會自動處理 Pinyin 轉換)
            output_path, tts_time = self.text_to_speech(translated_text, output_filename)

            processing_time = time.time() - start_time

            return translated_text, output_path, processing_time

        except Exception as e:
            logger.error(f"Error in translate and speak: {str(e)}")
            raise

# Global TTS service instance
tts_service = TTSService()
