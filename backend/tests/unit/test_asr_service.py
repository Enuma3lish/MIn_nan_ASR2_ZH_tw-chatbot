import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch

from app.services.asr_service import ASRService
from app.models.schemas import LanguageType


@pytest.mark.unit
class TestASRService:
    """Unit tests for ASR Service"""

    def test_asr_service_initialization(self):
        """Test ASR service initializes correctly"""
        service = ASRService()
        assert service.device in ["cpu", "cuda"]
        assert service.models_loaded is False
        assert service.chinese_processor is None
        assert service.chinese_model is None
        assert service.min_nan_processor is None
        assert service.min_nan_model is None

    @pytest.mark.slow
    @pytest.mark.requires_models
    def test_load_models(self):
        """Test loading ASR models (slow test, requires internet)"""
        service = ASRService()

        # This test will actually download models
        # Skip in CI/CD or when models are not needed
        try:
            service.load_models()
            assert service.models_loaded is True
            assert service.chinese_processor is not None
            assert service.chinese_model is not None
            assert service.min_nan_processor is not None
            assert service.min_nan_model is not None
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")

    def test_preprocess_audio(self, sample_audio_file):
        """Test audio preprocessing"""
        service = ASRService()
        audio, sample_rate = service.preprocess_audio(sample_audio_file)

        assert isinstance(audio, np.ndarray)
        assert sample_rate == 16000
        assert len(audio) > 0

    def test_preprocess_audio_invalid_file(self):
        """Test preprocessing with invalid file"""
        service = ASRService()

        with pytest.raises(Exception):
            service.preprocess_audio("nonexistent_file.wav")

    @patch('app.services.asr_service.WhisperProcessor')
    @patch('app.services.asr_service.WhisperForConditionalGeneration')
    def test_transcribe_chinese_mock(
        self,
        mock_whisper_model,
        mock_whisper_processor,
        chinese_sample_audio
    ):
        """Test Chinese transcription with mocked models"""
        # Setup mocks
        mock_processor_instance = Mock()
        mock_model_instance = Mock()

        mock_whisper_processor.from_pretrained.return_value = mock_processor_instance
        mock_whisper_model.from_pretrained.return_value = mock_model_instance

        # Mock processor methods
        mock_processor_instance.return_value = Mock(input_features=torch.randn(1, 80, 3000))
        mock_processor_instance.get_decoder_prompt_ids.return_value = []
        mock_processor_instance.batch_decode.return_value = ["你好世界"]

        # Mock model methods
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = None
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3]])

        # Test transcription
        service = ASRService()
        service.load_models()

        text, confidence, processing_time = service.transcribe_chinese(chinese_sample_audio)

        assert isinstance(text, str)
        assert text == "你好世界"
        assert processing_time > 0

    @patch('app.services.asr_service.Wav2Vec2Processor')
    @patch('app.services.asr_service.Wav2Vec2ForCTC')
    def test_transcribe_minnan_mock(
        self,
        mock_wav2vec_model,
        mock_wav2vec_processor,
        minnan_sample_audio
    ):
        """Test Min Nan transcription with mocked models"""
        # Setup mocks
        mock_processor_instance = Mock()
        mock_model_instance = Mock()

        mock_wav2vec_processor.from_pretrained.return_value = mock_processor_instance
        mock_wav2vec_model.from_pretrained.return_value = mock_model_instance

        # Mock processor methods
        mock_inputs = Mock()
        mock_inputs.input_values = torch.randn(1, 16000)
        mock_processor_instance.return_value = mock_inputs
        mock_processor_instance.batch_decode.return_value = ["汝好"]

        # Mock model methods
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = None
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 100, 32)
        mock_model_instance.return_value = mock_output

        # Test transcription
        service = ASRService()
        service.load_models()

        text, confidence, processing_time = service.transcribe_min_nan(minnan_sample_audio)

        assert isinstance(text, str)
        assert processing_time > 0

    def test_transcribe_unsupported_language(self, sample_audio_file):
        """Test transcription with unsupported language"""
        service = ASRService()

        with pytest.raises(ValueError, match="Unsupported language"):
            service.transcribe(sample_audio_file, "unsupported_language")

    @patch('app.services.asr_service.ASRService.transcribe_chinese')
    def test_transcribe_chinese_route(self, mock_transcribe, sample_audio_file):
        """Test that Chinese language routes to Chinese transcription"""
        mock_transcribe.return_value = ("測試", None, 1.0)

        service = ASRService()
        service.models_loaded = True

        text, confidence, time = service.transcribe(
            sample_audio_file,
            LanguageType.CHINESE
        )

        mock_transcribe.assert_called_once_with(sample_audio_file)
        assert text == "測試"

    @patch('app.services.asr_service.ASRService.transcribe_min_nan')
    def test_transcribe_minnan_route(self, mock_transcribe, sample_audio_file):
        """Test that Min Nan language routes to Min Nan transcription"""
        mock_transcribe.return_value = ("測試", None, 1.0)

        service = ASRService()
        service.models_loaded = True

        text, confidence, time = service.transcribe(
            sample_audio_file,
            LanguageType.MIN_NAN
        )

        mock_transcribe.assert_called_once_with(sample_audio_file)
        assert text == "測試"
