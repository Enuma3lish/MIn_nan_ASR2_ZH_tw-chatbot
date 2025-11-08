# Testing Guide

This guide covers how to run and write tests for the Min Nan & Chinese Voice Chatbot backend.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── test_asr_service.py
│   ├── test_tts_service.py
│   └── test_translation_service.py
└── integration/             # Integration tests
    └── test_api_endpoints.py
```

## Running Tests

### Install Test Dependencies

```bash
# Install all dependencies including testing
pip install -r requirements.txt

# Or install test dependencies separately
pip install -r requirements-test.txt
```

### Run All Tests

```bash
# Run all tests with coverage
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=app --cov-report=html
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only fast tests (exclude slow model-loading tests)
pytest -m "not slow"

# Run tests that require models
pytest -m requires_models
```

### Run Specific Test Files

```bash
# Run ASR service tests
pytest tests/unit/test_asr_service.py

# Run TTS service tests
pytest tests/unit/test_tts_service.py

# Run translation service tests
pytest tests/unit/test_translation_service.py

# Run API endpoint tests
pytest tests/integration/test_api_endpoints.py
```

### Run Specific Tests

```bash
# Run a specific test class
pytest tests/unit/test_asr_service.py::TestASRService

# Run a specific test function
pytest tests/unit/test_asr_service.py::TestASRService::test_asr_service_initialization
```

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.unit` - Unit tests (fast, mocked)
- `@pytest.mark.integration` - Integration tests (test full API)
- `@pytest.mark.slow` - Slow tests that download models
- `@pytest.mark.requires_models` - Tests that require AI models to be loaded

### Examples:

```python
@pytest.mark.unit
def test_something_fast():
    # Fast unit test
    pass

@pytest.mark.slow
@pytest.mark.requires_models
def test_with_real_models():
    # Slow test that loads actual models
    pass
```

## Test Fixtures

### Available Fixtures

- `test_data_dir` - Temporary directory for test data
- `sample_audio_file` - Generated sample WAV audio file
- `chinese_sample_audio` - Simulated Chinese speech audio
- `minnan_sample_audio` - Simulated Min Nan speech audio
- `app` - FastAPI test application
- `client` - Test client for API requests
- `mock_asr_service` - Mocked ASR service
- `mock_tts_service` - Mocked TTS service
- `sample_chinese_text` - Sample Chinese text
- `sample_minnan_text` - Sample Min Nan text

### Using Fixtures

```python
def test_with_audio_file(sample_audio_file):
    # Use the generated audio file
    with open(sample_audio_file, 'rb') as f:
        audio_data = f.read()
    assert len(audio_data) > 0
```

## Coverage Reports

After running tests with coverage:

```bash
pytest --cov=app --cov-report=html
```

View the HTML coverage report:

```bash
# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Writing Tests

### Unit Test Example

```python
import pytest
from app.services.translation_service import TranslationService
from app.models.schemas import LanguageType

@pytest.mark.unit
class TestTranslationService:
    def test_dictionary_translate(self):
        service = TranslationService()
        result = service.dictionary_translate(
            "你好",
            LanguageType.CHINESE,
            LanguageType.MIN_NAN
        )
        assert result == "汝好"
```

### Integration Test Example

```python
import pytest

@pytest.mark.integration
def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### Mocking Example

```python
from unittest.mock import patch

@patch('app.services.asr_service.WhisperProcessor')
def test_with_mock(mock_processor):
    # Mock processor behavior
    mock_processor.from_pretrained.return_value = Mock()
    # Test code here
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest --cov=app --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Test Best Practices

### 1. Use Mocks for External Dependencies

```python
# Good - mocked models
@patch('app.services.asr_service.WhisperProcessor')
def test_asr(mock_processor):
    pass

# Avoid - loading actual models in tests
def test_asr_slow():
    service = ASRService()
    service.load_models()  # Very slow!
```

### 2. Mark Slow Tests

```python
@pytest.mark.slow
@pytest.mark.requires_models
def test_with_real_models():
    # This test actually loads models
    pass
```

### 3. Use Fixtures for Common Setup

```python
@pytest.fixture
def configured_service():
    service = MyService()
    service.configure()
    return service

def test_something(configured_service):
    # Use the pre-configured service
    pass
```

### 4. Test Edge Cases

```python
def test_empty_input(service):
    result = service.process("")
    assert result is not None

def test_invalid_input(service):
    with pytest.raises(ValueError):
        service.process(None)
```

## Troubleshooting

### Tests Fail with Import Errors

Make sure you're in the backend directory:

```bash
cd backend
pytest
```

Or set PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

### Tests Fail with Model Download Errors

Skip slow tests that download models:

```bash
pytest -m "not slow"
```

### Tests Fail with Permission Errors

Check that temp directories can be created:

```bash
# Create necessary directories
mkdir -p uploads outputs model_cache
chmod 755 uploads outputs model_cache
```

## Performance Testing

### Benchmark Tests

```python
import time

def test_performance(sample_audio_file):
    start = time.time()
    # Run operation
    duration = time.time() - start
    assert duration < 5.0  # Should complete in under 5 seconds
```

## Test Data

### Generating Test Audio

```python
import numpy as np
from scipy.io import wavfile

def generate_test_audio(filename, duration=1, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
    audio = np.int16(audio * 32767)
    wavfile.write(filename, sample_rate, audio)
```

## Documentation

- Tests should be self-documenting with clear names
- Use docstrings to explain complex test scenarios
- Add comments for non-obvious assertions

```python
def test_chinese_to_minnan_translation():
    """
    Test translation from Chinese to Min Nan.

    Verifies that common greeting "你好" is correctly
    translated to Min Nan equivalent "汝好".
    """
    service = TranslationService()
    result, _ = service.translate(
        "你好",
        LanguageType.CHINESE,
        LanguageType.MIN_NAN
    )
    # "你好" (hello in Chinese) -> "汝好" (hello in Min Nan)
    assert result == "汝好"
```
