# Production-Ready Text-to-Speech AI API Platform

A high-performance, commercially viable Text-to-Speech (TTS) REST API microservice built on **FastAPI**, **PyTorch**, and **Kokoro-82M** (Apache 2.0).

---

## 📑 Table of Contents

- [1. Architecture Overview](#1-architecture-overview)
- [2. Model Selection & Licensing](#2-model-selection--licensing)
- [3. Features](#3-features)
- [4. Quickstart — Windows (PowerShell)](#4-quickstart--windows-powershell)
- [5. Quickstart — Linux / macOS](#5-quickstart--linux--macos)
- [6. Docker & Docker Compose Deployment](#6-docker--docker-compose-deployment)
- [7. API Documentation & cURL Examples](#7-api-documentation--curl-examples)
- [8. Client SDKs](#8-client-sdks)
  - [Python Client](#python-client)
  - [JavaScript / Node.js Client](#javascript--nodejs-client)
- [9. Configuration & Environment Variables](#9-configuration--environment-variables)
- [10. Testing & Benchmarking](#10-testing--benchmarking)
- [11. Observability & Monitoring](#11-observability--monitoring)
- [12. Security Threat Model & Mitigations](#12-security-threat-model--mitigations)
- [13. Troubleshooting & FAQ](#13-troubleshooting--faq)

---

## 1. Architecture Overview

The system strictly decouples the HTTP interface from ML inference and storage backends:

```
┌────────────────────────────────────────────────────────┐
│      Client (Web, Mobile, Python SDK, Node.js)        │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP / REST (X-API-Key, JSON)
                            ▼
┌────────────────────────────────────────────────────────┐
│                   API Layer (FastAPI)                  │
│  - Request ID Tracing (ULID)                           │
│  - Token Bucket Rate Limiting (SlowAPI / Redis)        │
│  - Strict Pydantic v2 Input Validation                │
│  - Error Sanitization & Centralized Exception Handling │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                   Service Layer                        │
│  - TTSService (Orchestrator & Concurrency Limiter)     │
│  - Text Chunking & Sentence Boundary Segmentation      │
│  - Deterministic SHA-256 Cache-Aside Engine            │
│  - Audio Normalization & Transcoding (WAV / MP3)       │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
               ▼                          ▼
┌───────────────────────────┐  ┌─────────────────────────┐
│     Engine Layer          │  │     Storage Layer       │
│  BaseTTSEngine (ABC)      │  │  StorageBackend (ABC)   │
│  └── KokoroEngine (82M)   │  │  ├── LocalStorage       │
│  └── PiperEngine (stub)   │  │  └── S3 / MinIO / R2    │
│  └── XTTSEngine (stub)    │  └─────────────────────────┘
└──────────────┬────────────┘
               │
               ▼
┌───────────────────────────┐
│   Hardware Acceleration   │
│   CUDA GPU / CPU Fallback │
└───────────────────────────┘
```

---

## 2. Model Selection & Licensing

### Why Kokoro-82M?
| Criteria | Kokoro-82M | XTTS-v2 | Piper | SpeechT5 |
| :--- | :--- | :--- | :--- | :--- |
| **Model License** | **Apache 2.0 (Commercial OK)** | CPML (Restricted) | MIT | CC-BY-NC |
| **Parameters** | 82 Million | ~460 Million | ~20 Million | ~150 Million |
| **Naturalness** | ⭐⭐⭐⭐⭐ (Near ElevenLabs) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Sample Rate** | **24,000 Hz** (Crisp) | 22,050 Hz | 22,050 Hz | 16,000 Hz |
| **Languages** | 8 Languages | 17+ Languages | 40+ Languages | 1 Language |
| **CPU Viability**| Excellent (<0.3x RTF) | Poor (>1.5x RTF) | Ultra-fast | Good |
| **GPU VRAM** | < 500 MB | > 4 GB | Negligible | < 1 GB |

### Licensing Inventory
- **Codebase License:** MIT License.
- **Model Weights:** Apache 2.0 (`hexgrad/Kokoro-82M`).
- **G2P Engine:** Apache 2.0 (`hexgrad/misaki`).
- **System Dependencies:** `espeak-ng` (GPL-3.0 invoked as external runtime CLI) and `ffmpeg` (LGPL-2.1+ invoked dynamically).

---

## 3. Features

- **High-Fidelity Audio:** 24,000 Hz studio-quality WAV and MP3 speech generation.
- **Synchronous & Asynchronous:** Stream audio immediately or submit batch jobs with webhook callbacks.
- **Deterministic Caching:** SHA-256 hash caching of identical synthesis requests.
- **Security & Multi-Tenancy:** Secure HMAC SHA-256 API key hashing, per-client usage tracking, and rate limiting.
- **Sentence-Aware Text Chunking:** Handles 50,000+ character long-form text without cutting words or sentences.
- **Prometheus Observability:** Native `/metrics` endpoint with latency histograms and real-time factors.

---

## 4. Quickstart — Windows (PowerShell)

### Prerequisites
- Python 3.11 or 3.12 (Python 3.13 also supported for testing)
- [FFmpeg](https://ffmpeg.org/) (optional, required for MP3 format)
- [espeak-ng for Windows](https://github.com/espeak-ng/espeak-ng/releases) (optional, required for non-English languages)

### 1. Clone & Set Up Environment
```powershell
git clone https://github.com/your-org/tts-api.git
cd tts-api

# Create and activate Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip tools
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install Dependencies
```powershell
# For CPU-only (standard development):
pip install -r requirements.txt

# For NVIDIA GPU (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Initialize Configuration
```powershell
Copy-Item .env.example .env
```

### 4. Create an API Key & Start Server
```powershell
# Create an initial API key
python scripts/create_api_key.py --name "Dev Admin" --admin

# Start the development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Visit the interactive Swagger API documentation at: **http://localhost:8000/docs**

---

## 5. Quickstart — Linux / macOS

```bash
# 1. Install system dependencies
# Ubuntu / Debian:
sudo apt-get update && sudo apt-get install -y ffmpeg espeak-ng libsndfile1

# macOS (Homebrew):
brew install ffmpeg espeak-ng libsndfile

# 2. Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure and start
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 6. Docker & Docker Compose Deployment

### CPU Stack (FastAPI + Redis + Postgres + Worker)
```bash
docker compose up -d --build
```

### NVIDIA GPU Stack (CUDA Accelerated)
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

---

## 7. API Documentation & cURL Examples

### 1. Health Check
```bash
curl -X GET http://localhost:8000/api/v1/health
```
*Response:*
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "version": "1.0.0",
  "uptime_seconds": 124.5
}
```

### 2. List Voices
```bash
curl -X GET "http://localhost:8000/api/v1/voices?language=en&gender=female"
```

### 3. Synchronous TTS (Download WAV Audio)
```bash
curl -X POST http://localhost:8000/api/v1/tts \
  -H "X-API-Key: test-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome to our production text to speech AI platform.",
    "voice": "af_sarah",
    "language": "en",
    "speed": 1.0,
    "format": "wav"
  }' \
  --output speech.wav
```

### 4. Synchronous TTS (JSON Response Mode)
```bash
curl -X POST http://localhost:8000/api/v1/tts \
  -H "X-API-Key: test-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Audio metadata mode.",
    "voice": "af_sarah",
    "return_json": true
  }'
```
*Response:*
```json
{
  "request_id": "01J7K...",
  "audio_id": "01J7M...",
  "audio_url": "/api/v1/audio/01J7M...",
  "duration_seconds": 1.84,
  "format": "wav",
  "voice": "af_sarah",
  "language": "en",
  "model_name": "kokoro",
  "model_version": "0.9.4",
  "processing_time_ms": 284
}
```

### 5. Asynchronous TTS Job
```bash
curl -X POST http://localhost:8000/api/v1/tts/async \
  -H "X-API-Key: test-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a very long text article that will be processed in the background worker queue...",
    "voice": "af_sarah",
    "language": "en"
  }'
```
*Response:*
```json
{
  "job_id": "job_01J7M...",
  "status": "queued",
  "estimated_wait_seconds": 4,
  "poll_url": "/api/v1/jobs/job_01J7M..."
}
```

---

## 8. Client SDKs

### Python Client

```python
from sdk.python.tts_client import TTSClient

# Initialize client
client = TTSClient(
    base_url="http://localhost:8000",
    api_key="your_api_key_here"
)

# 1. Synthesize and save audio directly to disk
client.synthesize_to_file(
    text="Hello from Python! The TTS API works seamlessly.",
    voice="af_sarah",
    language="en",
    output_path="output.wav"
)

# 2. Async long-text synthesis
job_id = client.synthesize_async(
    text="Long novel chapter or article...",
    voice="af_sarah"
)
result = client.poll_job(job_id)
print(f"Audio ready at: {result['audio_url']}")
```

### JavaScript / Node.js Client

```javascript
const { TTSClient } = require('./sdk/javascript/tts_client');

const client = new TTSClient({
    baseUrl: 'http://localhost:8000',
    apiKey: 'your_api_key_here',
});

async function run() {
    // Generate and save audio
    await client.synthesizeToFile({
        text: 'Hello from Node.js! Studio quality speech generation.',
        voice: 'af_sarah',
        language: 'en',
        outputPath: './output.wav',
    });
    console.log('Audio saved successfully!');
}

run();
```

---

## 9. Configuration & Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `APP_ENV` | `development` | `development` \| `staging` \| `production` |
| `TTS_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` |
| `MAX_TEXT_LENGTH` | `5000` | Max characters for sync `/tts` |
| `MAX_TEXT_LENGTH_ASYNC` | `50000` | Max characters for async `/tts/async` |
| `MAX_CONCURRENT_SYNTHESIS`| `2` | Max parallel inference threads |
| `CACHE_ENABLED` | `true` | Enable cache-aside audio reuse |
| `CACHE_BACKEND` | `redis` | `redis` \| `memory` \| `none` |
| `STORAGE_BACKEND` | `local` | `local` \| `s3` |
| `DATABASE_URL` | SQLite / Postgres | Async SQLAlchemy DB URL |
| `AUDIO_RETENTION_HOURS` | `24` | Auto-cleanup window for local files |

---

## 10. Testing & Benchmarking

### Automated Test Suite
```bash
# Run 30 comprehensive unit & integration tests
pytest -v
```

### Model Performance Benchmark
```bash
python scripts/benchmark.py --iterations 3 --output benchmark_results.json
```

---

## 11. Observability & Monitoring

- **Prometheus Metrics:** Scrape `http://localhost:8000/metrics`.
- **Metrics Tracked:**
  - `tts_api_http_requests_total`
  - `tts_api_http_request_duration_seconds`
  - `tts_api_synthesis_duration_seconds`
  - `tts_api_real_time_factor`
  - `tts_api_cache_hits_total` / `tts_api_cache_misses_total`
  - `tts_api_active_synthesis`
- **Structured Logs:** Full JSON logging with ULID `request_id`, duration, and status codes.

---

## 12. Security Threat Model & Mitigations

1. **Denial of Service (DoS) & Resource Exhaustion:**
   - **Mitigation:** Token bucket rate limiting (SlowAPI) + bounded concurrency semaphore (`MAX_CONCURRENT_SYNTHESIS=2`).
2. **Arbitrary File Upload & Path Traversal:**
   - **Mitigation:** Strict filename sanitization (`isalnum`), directory sandboxing, and no raw file upload paths accepted from users.
3. **Secret Leakage:**
   - **Mitigation:** HMAC SHA-256 one-way hashing for API keys; keys never printed in logs; Python stack traces intercepted and sanitized.
4. **Unauthorized Voice Cloning:**
   - **Mitigation:** `POST /api/v1/voices` requires affirmative consent verification (`consent_confirmed=true`) with full audit logging.

---

## 13. Troubleshooting & FAQ

- **Q: `ffmpeg` not found error when requesting MP3 format.**
  - **Fix:** Install FFmpeg on your operating system (`choco install ffmpeg` on Windows or `sudo apt install ffmpeg` on Linux).
- **Q: Model generation fails for non-English languages.**
  - **Fix:** Install `espeak-ng` system binary (`sudo apt install espeak-ng` or download Windows MSI).
- **Q: CUDA out of memory on low-end GPUs.**
  - **Fix:** Set `TTS_DEVICE=cpu` in `.env`. Kokoro-82M is lightweight and executes in real-time even on standard multi-core CPUs.
