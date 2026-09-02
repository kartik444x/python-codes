"""
app/core/metrics.py

Prometheus instrumentation for production observability.
Metrics exposed:
- Total HTTP requests counter by endpoint, method, status
- Request latency histogram
- TTS synthesis duration and RTF (Real-Time Factor) histograms
- Active synthesis gauge
- Cache hit / miss counters
- Storage operation counters
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from app.core.config import get_settings

settings = get_settings()
PREFIX = settings.metrics_prefix

# HTTP Request Metrics
HTTP_REQUESTS_TOTAL = Counter(
    f"{PREFIX}_http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    f"{PREFIX}_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# TTS Model Inference Metrics
TTS_SYNTHESIS_TOTAL = Counter(
    f"{PREFIX}_synthesis_total",
    "Total number of TTS synthesis operations",
    ["voice", "language", "status"],
)

TTS_SYNTHESIS_DURATION_SECONDS = Histogram(
    f"{PREFIX}_synthesis_duration_seconds",
    "TTS model inference time in seconds",
    ["voice", "language"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
)

TTS_AUDIO_DURATION_SECONDS = Histogram(
    f"{PREFIX}_generated_audio_duration_seconds",
    "Duration of generated audio in seconds",
    buckets=[1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

TTS_REAL_TIME_FACTOR = Histogram(
    f"{PREFIX}_real_time_factor",
    "Real-time factor (inference_time / audio_duration)",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
)

TTS_ACTIVE_SYNTHESIS = Gauge(
    f"{PREFIX}_active_synthesis",
    "Number of active in-flight synthesis tasks",
)

# Cache Metrics
CACHE_HITS_TOTAL = Counter(
    f"{PREFIX}_cache_hits_total",
    "Total number of audio cache hits",
    ["backend"],
)

CACHE_MISSES_TOTAL = Counter(
    f"{PREFIX}_cache_misses_total",
    "Total number of audio cache misses",
    ["backend"],
)

# Storage Metrics
STORAGE_OPERATIONS_TOTAL = Counter(
    f"{PREFIX}_storage_operations_total",
    "Total number of storage operations",
    ["operation", "backend", "status"],
)
