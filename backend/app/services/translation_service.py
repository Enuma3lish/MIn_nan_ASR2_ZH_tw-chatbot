import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import time
import logging
from typing import Tuple, Optional
from ..config import settings
from ..models.schemas import LanguageType

logger = logging.getLogger(__name__)


class TranslationService:
    """
    Service for translating between Chinese and Min Nan languages
    Uses Neural Machine Translation (NMT) with Facebook's M2M-100 model
    Falls back to dictionary-based translation for common phrases
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Translation Service initializing on device: {self.device}")

        # Initialize models as None - lazy loading
        self.tokenizer = None
        self.model = None
        self.models_loaded = False

        # Dictionary-based translation fallback for common phrases
        # Min Nan romanization (POJ - Pe̍h-ōe-jī) to Chinese mapping
        self.chinese_to_minnan_dict = {
            "你好": "汝好",  # Hello
            "早安": "早",  # Good morning
            "謝謝": "多謝",  # Thank you
            "再見": "再會",  # Goodbye
            "是": "是",  # Yes
            "不是": "毋是",  # No
            "好": "好",  # Good/OK
            "不好": "無好",  # Not good
            "吃飯": "食飯",  # Eat meal
            "喝水": "飲水",  # Drink water
            "什麼": "啥物",  # What
            "哪裡": "佗位",  # Where
            "怎麼": "按怎",  # How
            "誰": "啥人",  # Who
            "為什麼": "為啥物",  # Why
            "多少": "偌濟",  # How much/many
            "今天": "今仔日",  # Today
            "明天": "明仔載",  # Tomorrow
            "昨天": "昨昏",  # Yesterday
            "現在": "這馬",  # Now
            "這個": "這个",  # This
            "那個": "彼个",  # That
            "我": "我",  # I/me
            "你": "汝",  # You
            "他": "伊",  # He/him
            "她": "伊",  # She/her
            "我們": "阮",  # We/us
            "你們": "恁",  # You (plural)
            "他們": "𪜶",  # They/them
            "愛": "愛",  # Love
            "喜歡": "佮意",  # Like
            "想": "想欲",  # Want/think
            "知道": "知影",  # Know
            "不知道": "毋知",  # Don't know
            "很": "真",  # Very
            "非常": "誠",  # Very/extremely
            "一點": "一屑仔",  # A little
            "太": "傷",  # Too (much)
            "請": "請",  # Please
            "對不起": "歹勢",  # Sorry
            "沒關係": "無要緊",  # It's okay
            "歡迎": "歡迎",  # Welcome
            "加油": "拍拚",  # Keep it up/add oil
            "恭喜": "恭喜",  # Congratulations
            "新年快樂": "新年恭喜",  # Happy New Year
            "生日快樂": "生日快樂",  # Happy Birthday
            "平安": "平安",  # Peace/safe
            "健康": "健康",  # Health/healthy
        }

        # Reverse dictionary for Min Nan to Chinese
        self.minnan_to_chinese_dict = {
            v: k for k, v in self.chinese_to_minnan_dict.items()
        }

    def load_models(self):
        """Load NMT models (lazy loading)"""
        try:
            logger.info("Loading Translation models...")

            # Load M2M-100 model for multilingual translation
            # M2M-100 supports 100 languages including Chinese
            # We'll use it as a base and fine-tune or adapt for Min Nan
            model_name = "facebook/m2m100_418M"  # Smaller model for faster inference

            logger.info(f"Loading translation model: {model_name}")
            self.tokenizer = M2M100Tokenizer.from_pretrained(
                model_name,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            self.model = M2M100ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            self.model.to(self.device)
            self.model.eval()

            self.models_loaded = True
            logger.info("Translation models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading translation models: {str(e)}")
            logger.warning("Will use dictionary-based translation as fallback")
            # Don't raise - allow dictionary-based translation
            self.models_loaded = False

    def dictionary_translate(
        self,
        text: str,
        source_language: LanguageType,
        target_language: LanguageType
    ) -> str:
        """
        Dictionary-based translation for common phrases

        Args:
            text: Input text
            source_language: Source language
            target_language: Target language

        Returns:
            Translated text
        """
        if source_language == LanguageType.CHINESE and target_language == LanguageType.MIN_NAN:
            # Try exact match first
            if text in self.chinese_to_minnan_dict:
                return self.chinese_to_minnan_dict[text]

            # Try word-by-word translation
            translated_words = []
            words = list(text)  # Split into characters for Chinese
            for word in words:
                if word in self.chinese_to_minnan_dict:
                    translated_words.append(self.chinese_to_minnan_dict[word])
                else:
                    translated_words.append(word)  # Keep original if no translation
            return ''.join(translated_words)

        elif source_language == LanguageType.MIN_NAN and target_language == LanguageType.CHINESE:
            # Try exact match first
            if text in self.minnan_to_chinese_dict:
                return self.minnan_to_chinese_dict[text]

            # Try word-by-word translation
            translated_words = []
            words = list(text)
            for word in words:
                if word in self.minnan_to_chinese_dict:
                    translated_words.append(self.minnan_to_chinese_dict[word])
                else:
                    translated_words.append(word)
            return ''.join(translated_words)

        # No translation needed if same language
        return text

    def neural_translate(
        self,
        text: str,
        source_language: LanguageType,
        target_language: LanguageType
    ) -> str:
        """
        Neural Machine Translation using M2M-100

        Args:
            text: Input text
            source_language: Source language
            target_language: Target language

        Returns:
            Translated text
        """
        if not self.models_loaded:
            logger.warning("NMT models not loaded, using dictionary translation")
            return self.dictionary_translate(text, source_language, target_language)

        try:
            # Map our language types to M2M-100 language codes
            lang_code_map = {
                LanguageType.CHINESE: "zh",
                LanguageType.ZH_TW: "zh",
                # Note: M2M-100 doesn't have Min Nan, we'll use Chinese as approximation
                # In production, you would fine-tune the model for Min Nan
                LanguageType.MIN_NAN: "zh",
            }

            src_lang = lang_code_map.get(source_language, "zh")
            tgt_lang = lang_code_map.get(target_language, "zh")

            # Set source language
            self.tokenizer.src_lang = src_lang

            # Tokenize input
            encoded = self.tokenizer(text, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang),
                    max_length=512
                )

            # Decode translation
            translated = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )[0]

            return translated

        except Exception as e:
            logger.error(f"Error in neural translation: {str(e)}")
            logger.warning("Falling back to dictionary translation")
            return self.dictionary_translate(text, source_language, target_language)

    def translate(
        self,
        text: str,
        source_language: LanguageType,
        target_language: LanguageType,
        use_neural: bool = True
    ) -> Tuple[str, float]:
        """
        Translate text between languages

        Args:
            text: Input text
            source_language: Source language
            target_language: Target language
            use_neural: Whether to use neural translation (default: True)

        Returns:
            Tuple of (translated_text, processing_time)
        """
        start_time = time.time()

        logger.info(f"Translating from {source_language} to {target_language}")

        # If same language, no translation needed
        if source_language == target_language:
            return text, 0.0

        # Try dictionary translation first for common phrases (faster)
        dict_translation = self.dictionary_translate(
            text,
            source_language,
            target_language
        )

        # If dictionary translation changed the text significantly, use it
        if dict_translation != text and not use_neural:
            processing_time = time.time() - start_time
            return dict_translation, processing_time

        # Use neural translation if available and requested
        if use_neural and self.models_loaded:
            translated_text = self.neural_translate(
                text,
                source_language,
                target_language
            )
        else:
            translated_text = dict_translation

        # For Chinese to Min Nan, apply dictionary post-processing
        # This helps with domain-specific terms
        if source_language == LanguageType.CHINESE and target_language == LanguageType.MIN_NAN:
            # Replace known phrases in the translation
            for chinese, minnan in self.chinese_to_minnan_dict.items():
                if chinese in translated_text:
                    translated_text = translated_text.replace(chinese, minnan)

        processing_time = time.time() - start_time

        logger.info(
            f"Translation completed in {processing_time:.2f}s: "
            f"'{text}' -> '{translated_text}'"
        )

        return translated_text, processing_time


# Global translation service instance
translation_service = TranslationService()
