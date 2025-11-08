# Min Nan & Chinese Voice Chatbot

A comprehensive voice conversion and chatbot system supporting both Chinese (Mandarin/Taiwanese) and Min Nan (Taiwanese Hokkien) languages.

## 🌟 Features

### Backend (FastAPI)
- 🎤 **Speech-to-Text (ASR)**
  - Chinese ASR using OpenAI Whisper
  - Min Nan ASR using Facebook Wav2Vec2

- 🔊 **Text-to-Speech (TTS)**
  - Min Nan TTS using Facebook MMS-TTS
  - Chinese to Min Nan voice conversion

- 🔄 **Voice-to-Voice Conversion**
  - Convert Chinese voice to Min Nan voice
  - Convert Min Nan voice to text and back

- 📝 **Text Input to Voice**
  - Convert user text input to Min Nan speech

### Frontend (Coming Soon)
- React.js web application
- Real-time voice recording and playback
- Interactive chat interface
- File upload support

### Mobile App (Coming Soon)
- React Native application
- Cross-platform (iOS & Android)
- Native voice recording
- Offline capabilities

## 🏗️ Architecture

```
MIn_nan_ASR2_ZH_tw-chatbot/
├── backend/           # FastAPI backend service
│   ├── app/
│   │   ├── api/      # REST API endpoints
│   │   ├── models/   # Data models & schemas
│   │   ├── services/ # ASR & TTS services
│   │   └── utils/    # Utility functions
│   ├── uploads/      # Temporary audio uploads
│   ├── outputs/      # Generated audio files
│   └── README.md     # Backend documentation
├── frontend/         # React.js web app (TBD)
└── mobile/           # React Native app (TBD)
```

## 🚀 Quick Start

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```

5. **Run the server**
   ```bash
   python run.py
   ```

6. **Access API documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Using Docker

```bash
cd backend
docker-compose up -d
```

## 📚 API Usage Examples

### 1. Text to Min Nan Speech

```bash
curl -X POST "http://localhost:8000/api/v1/tts/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，歡迎使用閩南語語音系統",
    "source_language": "chinese",
    "target_language": "min_nan"
  }'
```

### 2. Chinese Voice to Min Nan Voice

```bash
curl -X POST "http://localhost:8000/api/v1/voice-conversion" \
  -F "audio_file=@chinese_audio.wav" \
  -F "source_language=chinese" \
  -F "target_language=min_nan"
```

### 3. Transcribe Audio to Text

```bash
curl -X POST "http://localhost:8000/api/v1/asr/transcribe" \
  -F "audio_file=@audio.wav" \
  -F "language=min_nan"
```

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI
- **ML/AI**:
  - Transformers (Hugging Face)
  - PyTorch
  - OpenAI Whisper
- **Audio**: librosa, soundfile
- **Models**:
  - `openai/whisper-large-v3` - Chinese ASR
  - `facebook/wav2vec2-large-xlsr-53` - Min Nan ASR
  - `facebook/mms-tts-nan` - Min Nan TTS

### Frontend (Planned)
- React.js
- Material-UI / Ant Design
- Web Audio API
- Axios for API calls

### Mobile (Planned)
- React Native
- Expo
- React Native Voice
- React Native Sound

## 📖 Documentation

- [Backend Documentation](./backend/README.md)
- Frontend Documentation (Coming Soon)
- Mobile Documentation (Coming Soon)

## 🔧 Development

### Prerequisites
- Python 3.8+
- Node.js 16+ (for frontend/mobile)
- npm or yarn
- (Optional) CUDA-compatible GPU

### Backend Development

See [backend/README.md](./backend/README.md) for detailed instructions.

### Testing

```bash
cd backend
python test_api.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- Facebook AI Research for MMS-TTS models
- OpenAI for Whisper ASR
- Hugging Face for model hosting and transformers library

## 📧 Contact

For questions or support, please open an issue on GitHub.
