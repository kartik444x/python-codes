# ─────────────────────────────────────────────────────────────
# Production Multi-Stage Dockerfile for TTS AI API
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# Install runtime system packages:
# - espeak-ng: Phonemization fallback for multilingual Kokoro TTS
# - ffmpeg: Audio conversion (WAV to MP3)
# - libsndfile1: Audio file I/O
# - curl: Container healthcheck probe
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create unprivileged user for production security
RUN useradd -u 10001 -m -s /bin/sh appuser

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ app/
COPY scripts/ scripts/
COPY models/ models/

# Setup directories with non-root permissions
RUN mkdir -p /app/generated_audio /app/logs /app/models && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/live || exit 1

# Production server entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
