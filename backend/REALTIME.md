# Real-time Voice Recognition

This guide covers how to use the real-time voice recognition features via WebSocket for streaming audio from your microphone.

## Overview

The real-time ASR (Automatic Speech Recognition) system allows you to:

- 🎤 **Stream audio directly from your microphone**
- 📝 **Get real-time transcriptions** as you speak
- 🔄 **Voice-to-voice conversion** with live translation
- ⚡ **Low latency** processing
- 🌐 **Works in browsers** and Python clients

## Features

### 1. Real-time ASR

- Continuous audio streaming via WebSocket
- Support for Chinese and Min Nan languages
- Optional interim results for faster feedback
- Automatic audio buffering and processing

### 2. Voice Chat with Translation

- Real-time voice-to-voice conversion
- Automatic translation between Chinese and Min Nan
- Generated audio output in target language

## WebSocket Endpoints

### 1. `/api/v1/ws/asr` - Real-time ASR

Stream audio and receive transcriptions in real-time.

### 2. `/api/v1/ws/voice-chat` - Voice Chat

Stream audio and receive translated audio output.

## Quick Start

### Option 1: Browser Client (Recommended for Testing)

1. **Start the backend server:**

```bash
cd backend
python run.py
```

2. **Open the browser client:**

```bash
# Open in your browser
open examples/realtime_asr_client.html
```

3. **Configure and start recording:**
   - Select language (Chinese or Min Nan)
   - Choose whether to show interim results
   - Click "Start Recording"
   - Grant microphone permission
   - Start speaking!

### Option 2: Python Client

1. **Install dependencies:**

```bash
pip install websockets pyaudio
```

2. **Run the client:**

```bash
cd backend/examples
python realtime_client.py
```

3. **Follow the prompts:**
   - Select language
   - Choose interim results option
   - Start speaking

## WebSocket Protocol

### Real-time ASR Protocol

#### Client → Server

**Configuration Message:**
```json
{
  "type": "config",
  "language": "chinese" | "min_nan",
  "interim_results": true | false
}
```

**Audio Data:**
- Send raw PCM audio bytes (Int16, 16kHz, mono)
- Recommended chunk size: 0.5-1.0 seconds

**Stop Recording:**
```json
{
  "type": "stop"
}
```

#### Server → Client

**Transcription Result:**
```json
{
  "type": "transcription",
  "text": "transcribed text",
  "is_final": true | false
}
```

**Error:**
```json
{
  "type": "error",
  "message": "error description"
}
```

**Stopped Confirmation:**
```json
{
  "type": "stopped"
}
```

### Voice Chat Protocol

#### Client → Server

**Configuration:**
```json
{
  "type": "config",
  "source_language": "chinese" | "min_nan",
  "target_language": "chinese" | "min_nan"
}
```

**Audio Data:**
- Send raw PCM audio bytes

**Stop and Process:**
```json
{
  "type": "stop"
}
```

#### Server → Client

**Transcription:**
```json
{
  "type": "transcription",
  "text": "transcribed text"
}
```

**Translation:**
```json
{
  "type": "translation",
  "text": "translated text"
}
```

**Audio Ready:**
```json
{
  "type": "audio_ready",
  "audio_url": "/api/v1/audio/filename.wav",
  "text": "translated text"
}
```

## Audio Format Requirements

- **Sample Rate**: 16000 Hz (16 kHz)
- **Bit Depth**: 16-bit
- **Channels**: 1 (mono)
- **Format**: PCM (raw audio)

## Examples

### JavaScript/Browser Example

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/asr');

ws.onopen = async () => {
    // Send configuration
    ws.send(JSON.stringify({
        type: 'config',
        language: 'chinese',
        interim_results: false
    }));

    // Start audio capture
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(stream);

    // Process audio chunks
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
        const audioData = e.inputBuffer.getChannelData(0);
        // Convert to Int16
        const int16Data = new Int16Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            int16Data[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32768));
        }
        ws.send(int16Data.buffer);
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);

    if (message.type === 'transcription') {
        console.log('Transcription:', message.text);
        console.log('Is Final:', message.is_final);
    }
};

// Stop recording
function stop() {
    ws.send(JSON.stringify({ type: 'stop' }));
}
```

### Python Example

```python
import asyncio
import websockets
import json
import pyaudio

async def stream_audio():
    # Connect to WebSocket
    async with websockets.connect('ws://localhost:8000/api/v1/ws/asr') as ws:
        # Send configuration
        await ws.send(json.dumps({
            'type': 'config',
            'language': 'chinese',
            'interim_results': False
        }))

        # Start audio capture
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8000  # 0.5 seconds
        )

        # Send audio and receive transcriptions
        async def send_audio():
            while True:
                data = stream.read(8000)
                await ws.send(data)
                await asyncio.sleep(0.01)

        async def receive_transcriptions():
            while True:
                message = await ws.recv()
                data = json.loads(message)
                if data['type'] == 'transcription':
                    print(f"Transcription: {data['text']}")

        # Run both tasks
        await asyncio.gather(
            send_audio(),
            receive_transcriptions()
        )

asyncio.run(stream_audio())
```

## Performance Optimization

### Client-side

1. **Audio Chunk Size**
   - Smaller chunks (0.5s): Lower latency, more network overhead
   - Larger chunks (1-2s): Higher latency, less overhead
   - Recommended: 0.5-1.0 seconds

2. **Buffer Management**
   - Pre-allocate audio buffers
   - Use typed arrays (Int16Array) in JavaScript
   - Minimize conversions

3. **Network**
   - Use WebSocket compression if available
   - Consider binary frames instead of JSON for audio

### Server-side

1. **Model Optimization**
   - Preload models on startup
   - Use GPU if available
   - Consider smaller/faster models for interim results

2. **Concurrency**
   - Each WebSocket connection runs in its own async task
   - Server can handle multiple simultaneous connections
   - Monitor server resources under load

## Troubleshooting

### Microphone Access Issues

**Browser:**
- Ensure HTTPS or localhost (HTTP allowed for localhost only)
- Grant microphone permission when prompted
- Check browser console for errors

**Python:**
- Ensure PyAudio is installed: `pip install pyaudio`
- On Linux, may need: `sudo apt-get install portaudio19-dev`
- On macOS: `brew install portaudio`
- On Windows: PyAudio binary wheels available via pip

### Connection Issues

**WebSocket Connection Failed:**
```
Error: Connection failed
```

**Solution:**
- Verify backend server is running
- Check URL (ws:// not http://)
- Verify port number (default: 8000)
- Check CORS settings if cross-origin

### Audio Quality Issues

**Poor Transcription Quality:**

1. **Check audio input:**
   - Test microphone in system settings
   - Reduce background noise
   - Speak clearly and at normal pace

2. **Check sample rate:**
   - Ensure 16kHz sample rate
   - Verify audio format (16-bit PCM)

3. **Network issues:**
   - Check for packet loss
   - Monitor WebSocket latency

### Latency Issues

**High Latency (>3 seconds):**

1. **Reduce chunk size**
   - Use smaller audio chunks (0.5s)
   - Enable interim results for faster feedback

2. **Server optimization**
   - Use GPU acceleration
   - Preload models
   - Check server CPU/memory usage

3. **Network optimization**
   - Check network latency
   - Use local deployment
   - Enable WebSocket compression

## Advanced Features

### Session Management

The server automatically manages sessions:
- Each WebSocket connection gets a unique session ID
- Audio buffers are maintained per session
- Sessions are cleaned up on disconnect

### Interim Results

Enable interim results for faster feedback:

```json
{
  "type": "config",
  "language": "chinese",
  "interim_results": true
}
```

**Benefits:**
- Faster visual feedback
- Better user experience
- Lower perceived latency

**Trade-offs:**
- Less accurate than final results
- More processing overhead
- More network traffic

### Multiple Languages

Switch languages mid-session:

```json
{
  "type": "config",
  "language": "min_nan"
}
```

## API Reference

### StreamingASRService

```python
from app.services.streaming_service import StreamingASRService

# Create service
service = StreamingASRService(
    sample_rate=16000,
    chunk_duration=1.0
)

# Add audio chunk
service.add_audio_chunk(audio_bytes)

# Get transcription
text = await service.transcribe_stream(
    audio_chunk,
    language=LanguageType.CHINESE,
    interim_results=False
)

# Get final result
final_text = await service.transcribe_final(
    language=LanguageType.CHINESE
)

# Reset buffer
service.reset_buffer()
```

### StreamingSessionManager

```python
from app.services.streaming_service import streaming_session_manager

# Create session
session = streaming_session_manager.create_session("session_id")

# Get session
session = streaming_session_manager.get_session("session_id")

# Remove session
streaming_session_manager.remove_session("session_id")
```

## Testing

### Manual Testing

1. **Browser Test:**
   ```bash
   # Start server
   python run.py

   # Open browser client
   open examples/realtime_asr_client.html
   ```

2. **Python Test:**
   ```bash
   # Install dependencies
   pip install websockets pyaudio

   # Run client
   python examples/realtime_client.py
   ```

### Automated Testing

```bash
# Test WebSocket endpoints
pytest tests/integration/test_websocket_endpoints.py -v
```

## Production Deployment

### Security Considerations

1. **Use WSS (WebSocket Secure)**
   ```python
   # With SSL certificate
   uvicorn app.main:app \
       --host 0.0.0.0 \
       --port 443 \
       --ssl-keyfile /path/to/key.pem \
       --ssl-certfile /path/to/cert.pem
   ```

2. **Authentication**
   - Add authentication middleware
   - Validate WebSocket connections
   - Implement rate limiting

3. **CORS Configuration**
   - Restrict allowed origins
   - Update `BACKEND_CORS_ORIGINS` in `.env`

### Scaling

1. **Horizontal Scaling**
   - Use load balancer with WebSocket support
   - Implement sticky sessions
   - Consider Redis for session management

2. **Vertical Scaling**
   - Use GPU instances for model inference
   - Increase worker processes
   - Optimize memory usage

### Monitoring

1. **Metrics to Track**
   - Active WebSocket connections
   - Transcription latency
   - Error rates
   - Server resource usage

2. **Logging**
   - Log all WebSocket events
   - Track session lifecycle
   - Monitor model performance

## Limitations

1. **Audio Format**
   - Only supports PCM 16-bit 16kHz mono
   - No automatic format conversion

2. **Language Support**
   - Currently: Chinese and Min Nan
   - Cannot mix languages in single session

3. **Buffer Size**
   - Maximum 100 chunks (~100 seconds)
   - Older audio is discarded

4. **Concurrent Connections**
   - Limited by server resources
   - Each connection uses memory for audio buffer
   - Each transcription uses GPU/CPU

## Future Enhancements

- [ ] Support for more audio formats
- [ ] Automatic language detection
- [ ] Speaker diarization
- [ ] Punctuation and formatting
- [ ] Confidence scores
- [ ] Real-time translation streaming
- [ ] Audio preprocessing (noise reduction)
- [ ] Adaptive chunk sizing
- [ ] Session persistence
- [ ] Connection recovery

## References

- [WebSocket API (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [FastAPI WebSocket](https://fastapi.tiangolo.com/advanced/websockets/)
- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [PyAudio Documentation](https://people.csail.mit.edu/hubert/pyaudio/docs/)

## Support

For issues or questions:
- Check the [Troubleshooting](#troubleshooting) section
- Review server logs
- Open an issue on GitHub
