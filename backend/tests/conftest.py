import pytest
import os
import numpy as np
from scipy.io import wavfile
from fastapi.testclient import TestClient
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sample_audio_file(test_data_dir):
    """Generate a sample WAV audio file for testing"""
    # Generate a simple sine wave audio
    sample_rate = 16000
    duration = 2  # seconds
    frequency = 440  # Hz (A note)

    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate sine wave
    audio_data = np.sin(2 * np.pi * frequency * t)

    # Convert to int16
    audio_data = np.int16(audio_data * 32767)

    # Save to file
    file_path = os.path.join(test_data_dir, "sample_audio.wav")
    wavfile.write(file_path, sample_rate, audio_data)

    return file_path


@pytest.fixture(scope="session")
def chinese_sample_audio(test_data_dir):
    """Generate a Chinese speech sample (simulated)"""
    sample_rate = 16000
    duration = 3

    # Generate more complex audio with multiple frequencies
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = (
        0.3 * np.sin(2 * np.pi * 300 * t) +
        0.3 * np.sin(2 * np.pi * 500 * t) +
        0.2 * np.sin(2 * np.pi * 700 * t) +
        0.2 * np.random.randn(len(t))  # Add some noise
    )

    audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

    file_path = os.path.join(test_data_dir, "chinese_sample.wav")
    wavfile.write(file_path, sample_rate, audio_data)

    return file_path


@pytest.fixture(scope="session")
def minnan_sample_audio(test_data_dir):
    """Generate a Min Nan speech sample (simulated)"""
    sample_rate = 16000
    duration = 3

    # Generate different audio pattern
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = (
        0.3 * np.sin(2 * np.pi * 250 * t) +
        0.3 * np.sin(2 * np.pi * 450 * t) +
        0.2 * np.sin(2 * np.pi * 650 * t) +
        0.2 * np.random.randn(len(t))
    )

    audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

    file_path = os.path.join(test_data_dir, "minnan_sample.wav")
    wavfile.write(file_path, sample_rate, audio_data)

    return file_path


@pytest.fixture
def app():
    """Create FastAPI test application"""
    from app.main import app
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_asr_service(mocker):
    """Mock ASR service to avoid loading heavy models"""
    mock_service = mocker.Mock()
    mock_service.models_loaded = True
    mock_service.transcribe.return_value = ("測試文本", None, 1.5)
    return mock_service


@pytest.fixture
def mock_tts_service(mocker):
    """Mock TTS service to avoid loading heavy models"""
    mock_service = mocker.Mock()
    mock_service.models_loaded = True
    mock_service.text_to_speech.return_value = ("/path/to/audio.wav", 1.2)
    mock_service.translate_and_speak.return_value = ("翻譯文本", "/path/to/audio.wav", 2.0)
    return mock_service


@pytest.fixture
def sample_chinese_text():
    """Sample Chinese text for testing"""
    return "你好，這是一個測試"


@pytest.fixture
def sample_minnan_text():
    """Sample Min Nan text for testing"""
    return "汝好，這是一个測試"


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, tmp_path):
    """Setup test environment variables"""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "model_cache"))

    # Create directories
    os.makedirs(tmp_path / "uploads", exist_ok=True)
    os.makedirs(tmp_path / "outputs", exist_ok=True)
    os.makedirs(tmp_path / "model_cache", exist_ok=True)
