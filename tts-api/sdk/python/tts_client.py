"""
sdk/python/tts_client.py

Lightweight Python Client SDK for consuming the TTS AI API.

Usage:
    from sdk.python.tts_client import TTSClient

    client = TTSClient(base_url="http://localhost:8000", api_key="YOUR_API_KEY")
    client.synthesize_to_file(
        text="Hello from my Python application!",
        voice="af_sarah",
        language="en",
        output_path="speech.wav"
    )
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


class TTSError(Exception):
    """Exception raised by TTSClient operations."""
    pass


class TTSClient:
    """
    Official Python SDK for the Production TTS AI Platform.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            self._headers["X-API-Key"] = self.api_key

    def check_health(self) -> Dict[str, Any]:
        """Check API service health status."""
        url = f"{self.base_url}/api/v1/health"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=self._headers)
            if resp.status_code != 200:
                raise TTSError(f"Health check failed with status {resp.status_code}: {resp.text}")
            return resp.json()

    def list_voices(self, language: Optional[str] = None, gender: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available TTS voices."""
        url = f"{self.base_url}/api/v1/voices"
        params = {}
        if language:
            params["language"] = language
        if gender:
            params["gender"] = gender

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=self._headers, params=params)
            if resp.status_code != 200:
                raise TTSError(f"Listing voices failed: {resp.text}")
            return resp.json().get("voices", [])

    def synthesize(
        self,
        text: str,
        voice: str = "af_sarah",
        language: str = "en",
        speed: float = 1.0,
        format_ext: str = "wav",
    ) -> bytes:
        """
        Synthesize speech synchronously and return audio bytes directly.
        """
        url = f"{self.base_url}/api/v1/tts"
        payload = {
            "text": text,
            "voice": voice,
            "language": language,
            "speed": speed,
            "format": format_ext,
            "return_json": False,
        }
        headers = {**self._headers, "Accept": f"audio/{format_ext}"}

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                raise TTSError(f"Synthesis failed with HTTP {resp.status_code}: {resp.text}")
            return resp.content

    def synthesize_to_file(
        self,
        text: str,
        output_path: str | Path,
        voice: str = "af_sarah",
        language: str = "en",
        speed: float = 1.0,
        format_ext: str = "wav",
    ) -> Path:
        """
        Synthesize speech and write audio to a local file.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        audio_bytes = self.synthesize(
            text=text,
            voice=voice,
            language=language,
            speed=speed,
            format_ext=format_ext,
        )
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        return out_path

    def synthesize_async(
        self,
        text: str,
        voice: str = "af_sarah",
        language: str = "en",
        speed: float = 1.0,
        format_ext: str = "wav",
        webhook_url: Optional[str] = None,
    ) -> str:
        """
        Submit a long-form TTS job for background synthesis. Returns job_id.
        """
        url = f"{self.base_url}/api/v1/tts/async"
        payload = {
            "text": text,
            "voice": voice,
            "language": language,
            "speed": speed,
            "format": format_ext,
            "webhook_url": webhook_url,
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=self._headers)
            if resp.status_code not in (200, 202):
                raise TTSError(f"Async synthesis submission failed: {resp.text}")
            return resp.json()["job_id"]

    def poll_job(self, job_id: str, poll_interval: float = 1.0, max_wait: float = 120.0) -> Dict[str, Any]:
        """
        Poll job status until completion or failure.
        """
        url = f"{self.base_url}/api/v1/jobs/{job_id}"
        start_time = time.time()

        with httpx.Client(timeout=self.timeout) as client:
            while time.time() - start_time < max_wait:
                resp = client.get(url, headers=self._headers)
                if resp.status_code != 200:
                    raise TTSError(f"Polling job failed: {resp.text}")
                data = resp.json()
                if data["status"] == "completed":
                    return data
                elif data["status"] in ("failed", "cancelled"):
                    raise TTSError(f"Job {job_id} terminated with status: {data['status']} - Error: {data.get('error_message')}")
                time.sleep(poll_interval)

        raise TTSError(f"Job {job_id} timed out after {max_wait} seconds.")

    def download_audio(self, audio_id: str, output_path: str | Path, format_ext: str = "wav") -> Path:
        """
        Download previously synthesized audio file by audio_id.
        """
        url = f"{self.base_url}/api/v1/audio/{audio_id}?format={format_ext}&download=true"
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=self._headers)
            if resp.status_code != 200:
                raise TTSError(f"Downloading audio failed: {resp.text}")
            with open(out_path, "wb") as f:
                f.write(resp.content)
        return out_path
