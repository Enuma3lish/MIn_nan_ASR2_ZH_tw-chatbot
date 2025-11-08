#!/usr/bin/env python3
"""
Simple test script for the API endpoints
"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 80)


def test_tts():
    """Test text-to-speech endpoint"""
    print("Testing TTS endpoint...")
    data = {
        "text": "你好，這是一個測試",
        "source_language": "chinese",
        "target_language": "min_nan"
    }
    response = requests.post(f"{BASE_URL}/tts/synthesize", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    if response.status_code == 200:
        audio_url = response.json()["audio_url"]
        print(f"\nAudio file available at: http://localhost:8000{audio_url}")
    print("-" * 80)


def test_asr():
    """Test ASR endpoint with a sample audio file"""
    print("Testing ASR endpoint...")
    print("Note: You need to provide a valid audio file path")

    # Replace with actual audio file path
    audio_file_path = "path/to/your/audio.wav"

    try:
        with open(audio_file_path, 'rb') as f:
            files = {'audio_file': f}
            data = {'language': 'chinese'}
            response = requests.post(
                f"{BASE_URL}/asr/transcribe",
                files=files,
                data=data
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
    except FileNotFoundError:
        print(f"Audio file not found: {audio_file_path}")
        print("Please provide a valid audio file to test ASR")
    print("-" * 80)


def test_load_models():
    """Test model loading endpoint"""
    print("Testing model loading endpoint...")
    response = requests.post(f"{BASE_URL}/models/load")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 80)


if __name__ == "__main__":
    print("=" * 80)
    print("API Test Suite")
    print("=" * 80)
    print("\nMake sure the server is running on http://localhost:8000")
    print()

    try:
        test_health()
        test_load_models()
        test_tts()
        # test_asr()  # Uncomment and provide audio file path to test

        print("\n✅ Tests completed!")

    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the server.")
        print("Please make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
