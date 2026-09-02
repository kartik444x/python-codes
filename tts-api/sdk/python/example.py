"""
sdk/python/example.py

Quickstart usage example for the Python TTS Client SDK.
"""
from tts_client import TTSClient


def main():
    # 1. Initialize Client
    client = TTSClient(
        base_url="http://localhost:8000",
        api_key="test-key-123",
    )

    print("Checking service health...")
    health = client.check_health()
    print(f"Service status: {health['status']} (Model: {health.get('device')})")

    print("\nListing available voices...")
    voices = client.list_voices(language="en")
    print(f"Found {len(voices)} English voices. Primary: {voices[0]['name']} ({voices[0]['id']})")

    # 2. Synchronous Speech Generation
    print("\nSynthesizing speech synchronously...")
    output_file = client.synthesize_to_file(
        text="Hello! This audio was synthesized using the production Python client SDK.",
        voice="af_sarah",
        language="en",
        speed=1.0,
        format_ext="wav",
        output_path="sdk_output.wav",
    )
    print(f"✓ Speech saved to: {output_file}")


if __name__ == "__main__":
    main()
