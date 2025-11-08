# Min Nan & Chinese Voice Chatbot - Backend API

FastAPI backend for Chinese and Min Nan voice-to-text and text-to-voice conversion using Facebook's MMS models.

## Features

- 🎤 **Automatic Speech Recognition (ASR)**
  - Chinese speech-to-text using OpenAI Whisper
  - Min Nan speech-to-text using Wav2Vec2

- 🔊 **Text-to-Speech (TTS)**
  - Min Nan text-to-speech using Facebook MMS-TTS
  - Support for Chinese to Min Nan conversion

- 🔄 **Voice-to-Voice Conversion**
  - Convert Chinese voice to Min Nan voice
  - Convert Min Nan voice to text and back to voice

- 🌐 **RESTful API**
  - Well-documented OpenAPI/Swagger interface
  - CORS support for frontend integration
  - File upload and download capabilities

## Tech Stack

- **Framework**: FastAPI 0.104+
- **ML Libraries**:
  - Transformers (Hugging Face)
  - PyTorch
  - Whisper (OpenAI)
- **Audio Processing**: librosa, soundfile
- **Server**: Uvicorn (ASGI)

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster inference

## Installation

### 1. Clone the repository

```bash
cd backend
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` file to customize settings if needed.

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # API endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── asr_service.py   # Speech-to-text service
│   │   └── tts_service.py   # Text-to-speech service
│   └── utils/
│       └── __init__.py
├── uploads/                 # Temporary uploaded files
├── outputs/                 # Generated audio files
├── model_cache/            # Cached AI models
├── requirements.txt
├── .env.example
├── .gitignore
├── run.py                  # Application entry point
└── README.md
```

## Running the Server

### Development Mode (with auto-reload)

```bash
python run.py
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc

## API Endpoints

### Health Check

```http
GET /api/v1/health
```

Check API status and model loading status.

### Transcribe Audio (ASR)

```http
POST /api/v1/asr/transcribe
Content-Type: multipart/form-data

Parameters:
- audio_file: Audio file (wav, mp3, m4a, ogg, flac)
- language: "chinese" | "min_nan" | "zh_tw"
```

**Example using curl:**

```bash
curl -X POST "http://localhost:8000/api/v1/asr/transcribe" \
  -F "audio_file=@path/to/audio.wav" \
  -F "language=chinese"
```

**Response:**

```json
{
  "text": "你好世界",
  "language": "chinese",
  "confidence": null,
  "processing_time": 2.34
}
```

### Text-to-Speech (TTS)

```http
POST /api/v1/tts/synthesize
Content-Type: application/json

{
  "text": "你好",
  "source_language": "chinese",
  "target_language": "min_nan"
}
```

**Example using curl:**

```bash
curl -X POST "http://localhost:8000/api/v1/tts/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好",
    "source_language": "chinese",
    "target_language": "min_nan"
  }'
```

**Response:**

```json
{
  "audio_url": "/api/v1/audio/tts_abc123.wav",
  "text": "你好",
  "source_language": "chinese",
  "target_language": "min_nan",
  "processing_time": 1.56
}
```

### Voice-to-Voice Conversion

```http
POST /api/v1/voice-conversion
Content-Type: multipart/form-data

Parameters:
- audio_file: Audio file
- source_language: "chinese" | "min_nan" (default: "chinese")
- target_language: "chinese" | "min_nan" (default: "min_nan")
```

**Example using curl:**

```bash
curl -X POST "http://localhost:8000/api/v1/voice-conversion" \
  -F "audio_file=@path/to/chinese_audio.wav" \
  -F "source_language=chinese" \
  -F "target_language=min_nan"
```

**Response:**

```json
{
  "transcribed_text": "你好世界",
  "audio_url": "/api/v1/audio/converted_xyz789.wav",
  "source_language": "chinese",
  "target_language": "min_nan",
  "processing_time": 3.90
}
```

### Download Audio File

```http
GET /api/v1/audio/{filename}
```

Download a generated audio file.

### Preload Models

```http
POST /api/v1/models/load
```

Manually trigger loading of AI models (useful for warming up the server).

## Configuration

Edit the `.env` file to customize:

- **Model Selection**: Choose different Hugging Face models
- **CORS Origins**: Add your frontend URLs
- **Upload Settings**: Adjust file size limits and allowed formats
- **Device**: Use "cuda" for GPU acceleration or "cpu" for CPU-only

## Model Information

### ASR Models

1. **Chinese ASR**: `openai/whisper-large-v3`
   - High-quality Chinese speech recognition
   - Supports Traditional and Simplified Chinese

2. **Min Nan ASR**: `facebook/wav2vec2-large-xlsr-53`
   - Multilingual model that can be fine-tuned for Min Nan
   - Note: For better Min Nan support, consider fine-tuning this model

### TTS Models

1. **Min Nan TTS**: `facebook/mms-tts-nan`
   - Facebook's Massively Multilingual Speech TTS
   - Native support for Min Nan language

## Performance Optimization

### GPU Acceleration

To use GPU acceleration:

1. Install CUDA-compatible PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. Set `DEVICE=cuda` in `.env`

### Model Preloading

Uncomment the model preloading code in `app/main.py` startup event to load models on server start:

```python
@app.on_event("startup")
async def startup_event():
    from .services.asr_service import asr_service
    from .services.tts_service import tts_service
    logger.info("Preloading AI models...")
    asr_service.load_models()
    tts_service.load_models()
```

## Troubleshooting

### Models not loading

- Check internet connection (models download from Hugging Face)
- Verify sufficient disk space for model cache
- Check MODEL_CACHE_DIR permissions

### Out of memory errors

- Reduce batch size or use smaller models
- Use CPU instead of GPU
- Close other applications

### Audio format errors

- Ensure audio files are in supported formats (wav, mp3, m4a, ogg, flac)
- Check sample rate (16kHz recommended)
- Verify files are not corrupted

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Code Formatting

```bash
# Install formatting tools
pip install black isort

# Format code
black .
isort .
```

## API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.
