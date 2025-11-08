#!/usr/bin/env python3
"""
Python client for testing real-time ASR via WebSocket

This script captures audio from your microphone and sends it to the
WebSocket endpoint for real-time transcription.

Requirements:
    pip install websockets pyaudio
"""

import asyncio
import websockets
import json
import pyaudio
import sys


class RealtimeASRClient:
    """Client for real-time ASR via WebSocket"""

    def __init__(
        self,
        ws_url="ws://localhost:8000/api/v1/ws/asr",
        language="chinese",
        interim_results=False,
        sample_rate=16000,
        chunk_duration=0.5
    ):
        """
        Initialize client

        Args:
            ws_url: WebSocket URL
            language: Language for ASR (chinese or min_nan)
            interim_results: Whether to show interim results
            sample_rate: Audio sample rate (default: 16000 Hz)
            chunk_duration: Duration of each audio chunk in seconds
        """
        self.ws_url = ws_url
        self.language = language
        self.interim_results = interim_results
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)

        # PyAudio
        self.audio = None
        self.stream = None

        # WebSocket
        self.websocket = None

        print(f"Real-time ASR Client initialized")
        print(f"  WebSocket URL: {ws_url}")
        print(f"  Language: {language}")
        print(f"  Interim results: {interim_results}")
        print(f"  Sample rate: {sample_rate} Hz")

    async def connect(self):
        """Connect to WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            print("✅ Connected to server")

            # Send configuration
            config = {
                "type": "config",
                "language": self.language,
                "interim_results": self.interim_results
            }
            await self.websocket.send(json.dumps(config))

            # Wait for config confirmation
            response = await self.websocket.recv()
            response_data = json.loads(response)
            if response_data.get("type") == "config_updated":
                print(f"✅ Configuration updated: {response_data}")
            else:
                print(f"⚠️  Unexpected response: {response_data}")

        except Exception as e:
            print(f"❌ Error connecting: {str(e)}")
            raise

    async def start_audio_capture(self):
        """Start capturing audio from microphone"""
        try:
            self.audio = pyaudio.PyAudio()

            # Open stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            print("🎤 Microphone opened, starting capture...")
            print("Press Ctrl+C to stop")

        except Exception as e:
            print(f"❌ Error opening microphone: {str(e)}")
            raise

    async def send_audio(self):
        """Send audio chunks to server"""
        try:
            while True:
                # Read audio chunk
                audio_data = self.stream.read(
                    self.chunk_size,
                    exception_on_overflow=False
                )

                # Send to server
                await self.websocket.send(audio_data)

                # Small delay
                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        except Exception as e:
            print(f"❌ Error sending audio: {str(e)}")
            raise

    async def receive_messages(self):
        """Receive and process messages from server"""
        try:
            while True:
                message = await self.websocket.recv()
                data = json.loads(message)

                if data["type"] == "transcription":
                    is_final = data.get("is_final", True)
                    text = data.get("text", "")

                    if is_final:
                        print(f"\n✅ Final: {text}")
                    else:
                        print(f"\r⏳ Interim: {text}", end="", flush=True)

                elif data["type"] == "error":
                    print(f"\n❌ Error: {data['message']}")

                elif data["type"] == "stopped":
                    print("\n✅ Session stopped")
                    break

        except websockets.exceptions.ConnectionClosed:
            print("\n⚠️  Connection closed")
        except Exception as e:
            print(f"\n❌ Error receiving: {str(e)}")

    async def stop(self):
        """Stop recording and get final result"""
        if self.websocket:
            try:
                # Send stop message
                await self.websocket.send(json.dumps({"type": "stop"}))
                print("\n📝 Getting final transcription...")

                # Wait for final result
                await asyncio.sleep(2)

            except Exception as e:
                print(f"❌ Error stopping: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        # Stop audio stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        # Terminate PyAudio
        if self.audio:
            self.audio.terminate()

        # Close WebSocket
        if self.websocket:
            await self.websocket.close()

        print("✅ Cleanup complete")

    async def run(self):
        """Run the client"""
        try:
            # Connect
            await self.connect()

            # Start audio capture
            await self.start_audio_capture()

            # Create tasks for sending and receiving
            send_task = asyncio.create_task(self.send_audio())
            recv_task = asyncio.create_task(self.receive_messages())

            # Wait for either task to complete (or Ctrl+C)
            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            # Stop and get final result
            await self.stop()

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        finally:
            await self.cleanup()


async def main():
    """Main function"""
    print("=" * 60)
    print("  Min Nan & Chinese Real-time ASR Client")
    print("=" * 60)
    print()

    # Get configuration from user
    print("Select language:")
    print("  1. Chinese (中文)")
    print("  2. Min Nan (閩南語)")

    choice = input("Enter choice (1 or 2, default: 1): ").strip() or "1"

    language = "chinese" if choice == "1" else "min_nan"

    interim_input = input("Show interim results? (y/n, default: n): ").strip().lower()
    interim_results = interim_input == "y"

    print()

    # Create and run client
    client = RealtimeASRClient(
        language=language,
        interim_results=interim_results
    )

    await client.run()

    print()
    print("=" * 60)
    print("  Session ended")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)
