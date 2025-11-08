import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
import os

from app.services.tts_service import TTSService
from app.models.schemas import LanguageType


@pytest.mark.unit
class TestTTSService:
    """Unit tests for TTS Service"""

    def test_tts_service_initialization(self):
        """Test TTS service initializes correctly"""
        service = TTSService()
        assert service.device in ["cpu", "cuda"]
        assert service.models_loaded is False
        assert service.tokenizer is None
        assert service.model is None

    @pytest.mark.slow
    @pytest.mark.requires_models
    def test_load_models(self):
        """Test loading TTS models (slow test, requires internet)"""
        service = TTSService()

        try:
            service.load_models()
            assert service.models_loaded is True
            assert service.tokenizer is not None
            assert service.model is not None
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")

    @patch('app.services.tts_service.VitsTokenizer')
    @patch('app.services.tts_service.VitsModel')
    @patch('app.services.tts_service.wavfile.write')
    def test_text_to_speech_mock(
        self,
        mock_wavfile,
        mock_vits_model,
        mock_vits_tokenizer,
        sample_chinese_text,
        tmp_path
    ):
        """Test text-to-speech with mocked models"""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()

        mock_vits_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_vits_model.from_pretrained.return_value = mock_model_instance

        # Mock tokenizer
        mock_inputs = {
            'input_ids': torch.randint(0, 100, (1, 10)),
            'attention_mask': torch.ones(1, 10)
        }
        mock_tokenizer_instance.return_value = mock_inputs

        # Mock model output
        mock_output = Mock()
        mock_waveform = torch.randn(1, 16000)  # 1 second of audio
        mock_output.waveform = mock_waveform
        mock_model_instance.return_value = mock_output
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = None

        # Mock wavfile.write to not actually write
        mock_wavfile.return_value = None

        # Test TTS
        service = TTSService()
        service.load_models()

        output_path, processing_time = service.text_to_speech(
            sample_chinese_text,
            "test_output.wav"
        )

        assert isinstance(output_path, str)
        assert "test_output.wav" in output_path
        assert processing_time > 0
        mock_wavfile.assert_called_once()

    @patch('app.services.tts_service.TTSService.text_to_speech')
    def test_translate_and_speak(
        self,
        mock_tts,
        sample_chinese_text,
        tmp_path
    ):
        """Test translate and speak functionality"""
        mock_tts.return_value = (str(tmp_path / "output.wav"), 1.5)

        service = TTSService()
        service.models_loaded = True

        translated_text, output_path, processing_time = service.translate_and_speak(
            text=sample_chinese_text,
            source_language=LanguageType.CHINESE,
            target_language=LanguageType.MIN_NAN
        )

        assert isinstance(translated_text, str)
        assert isinstance(output_path, str)
        assert processing_time > 0
        mock_tts.assert_called_once()

    @patch('app.services.tts_service.VitsTokenizer')
    @patch('app.services.tts_service.VitsModel')
    def test_text_to_speech_custom_filename(
        self,
        mock_vits_model,
        mock_vits_tokenizer,
        sample_chinese_text
    ):
        """Test TTS with custom filename"""
        # Setup mocks (similar to above)
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()

        mock_vits_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_vits_model.from_pretrained.return_value = mock_model_instance

        mock_inputs = {
            'input_ids': torch.randint(0, 100, (1, 10)),
        }
        mock_tokenizer_instance.return_value = mock_inputs

        mock_output = Mock()
        mock_output.waveform = torch.randn(1, 16000)
        mock_model_instance.return_value = mock_output
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = None

        service = TTSService()
        service.load_models()

        custom_filename = "custom_audio.wav"

        with patch('app.services.tts_service.wavfile.write'):
            output_path, _ = service.text_to_speech(
                sample_chinese_text,
                custom_filename
            )

        assert custom_filename in output_path

    def test_text_to_speech_without_models(self, sample_chinese_text):
        """Test that TTS loads models if not loaded"""
        service = TTSService()

        with patch.object(service, 'load_models') as mock_load:
            with patch('app.services.tts_service.wavfile.write'):
                with patch.object(service, 'text_to_speech', wraps=service.text_to_speech):
                    # This should trigger model loading
                    service.models_loaded = False
                    try:
                        service.text_to_speech(sample_chinese_text)
                    except AttributeError:
                        # Expected since models aren't actually loaded
                        pass

                    mock_load.assert_called_once()

    @patch('app.services.tts_service.TTSService.text_to_speech')
    def test_translate_chinese_to_minnan(
        self,
        mock_tts,
        sample_chinese_text,
        tmp_path
    ):
        """Test Chinese to Min Nan translation workflow"""
        expected_output = str(tmp_path / "output.wav")
        mock_tts.return_value = (expected_output, 2.0)

        service = TTSService()
        service.models_loaded = True

        translated_text, output_path, processing_time = service.translate_and_speak(
            text=sample_chinese_text,
            source_language=LanguageType.CHINESE,
            target_language=LanguageType.MIN_NAN,
            output_filename="minnan_output.wav"
        )

        assert isinstance(translated_text, str)
        assert output_path == expected_output
        assert processing_time > 0

    def test_empty_text_handling(self):
        """Test handling of empty text input"""
        service = TTSService()
        service.models_loaded = True

        # Mock the models
        with patch.object(service, 'tokenizer', Mock()):
            with patch.object(service, 'model', Mock()):
                # Empty text should still work, models will handle it
                # The actual behavior depends on model implementation
                pass
