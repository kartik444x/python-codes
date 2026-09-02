"""
scripts/benchmark.py

Comprehensive TTS Inference and API Benchmarking Tool.

Measures:
- Cold-start & model loading time
- Warm inference latency across different text lengths
- Real-Time Factor (RTF = inference_time / audio_duration)
- Throughput (characters per second and audio seconds per second)
- Hardware utilization and memory footprint

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --iterations 5 --output benchmark_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_PROMPTS = [
    {
        "type": "Short (Notification / UI Prompt)",
        "text": "Your order has been processed and is ready for pickup.",
        "chars": 54,
    },
    {
        "type": "Medium (Paragraph / Chat Response)",
        "text": (
            "Artificial intelligence is transforming how software systems interact with human users. "
            "Modern text to speech models synthesize natural intonation, rhythm, and emotional nuance."
        ),
        "chars": 194,
    },
    {
        "type": "Long (Article / Reading)",
        "text": (
            "The architecture of modern deep learning voice synthesis pipelines incorporates acoustic "
            "modeling, neural vocoding, and grapheme to phoneme converters. By maintaining low latency "
            "and high real-time factors, edge and cloud servers can stream interactive speech seamlessly "
            "to mobile applications, browsers, and embedded hardware without noticeable delay."
        ),
        "chars": 387,
    },
]


def get_memory_mb() -> float:
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def run_benchmark(iterations: int = 3, voice: str = "af_sarah", language: str = "en") -> dict:
    from app.core.config import get_settings
    from app.engines import create_engine

    settings = get_settings()
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": settings.tts_model.value,
        "device": settings.resolved_tts_device,
        "iterations": iterations,
        "cold_start_seconds": 0.0,
        "prompts": [],
    }

    print("=" * 70)
    print(f"  TTS Model Performance Benchmark — {settings.tts_model.value.upper()}")
    print(f"  Device: {settings.resolved_tts_device.upper()} | Iterations: {iterations}")
    print("=" * 70)

    # 1. Cold Start
    mem_before = get_memory_mb()
    t0 = time.perf_counter()
    engine = create_engine()
    engine.load()
    cold_start_time = time.perf_counter() - t0
    mem_after = get_memory_mb()

    results["cold_start_seconds"] = round(cold_start_time, 3)
    results["memory_mb"] = round(mem_after - mem_before, 1) if mem_after > 0 else "N/A"

    print(f"\n[1] Model Loading Time: {cold_start_time:.2f}s (RAM delta: {results['memory_mb']} MB)")

    # 2. Warmup
    engine.warmup()
    print("[2] Engine warm-up completed.")

    # 3. Benchmark across test prompts
    print("\n[3] Running Inference Benchmarks:")
    print("-" * 70)
    print(f"{'Prompt Type':<35} | {'Audio(s)':<8} | {'Infer(s)':<8} | {'RTF':<6} | {'Status'}")
    print("-" * 70)

    for item in BENCHMARK_PROMPTS:
        p_type = item["type"]
        text = item["text"]
        chars = len(text)

        durations = []
        inference_times = []
        rtfs = []

        for _ in range(iterations):
            t_start = time.perf_counter()
            res = engine.synthesize(text=text, voice_id=voice, language=language, speed=1.0)
            t_infer = time.perf_counter() - t_start

            audio_dur = res.duration_seconds
            rtf = t_infer / audio_dur if audio_dur > 0 else 0

            durations.append(audio_dur)
            inference_times.append(t_infer)
            rtfs.append(rtf)

        avg_audio = float(np.mean(durations))
        avg_infer = float(np.mean(inference_times))
        avg_rtf = float(np.mean(rtfs))
        char_per_sec = chars / avg_infer if avg_infer > 0 else 0

        status_str = "FAST (<0.5x)" if avg_rtf < 0.5 else ("REALTIME" if avg_rtf <= 1.0 else "SLOW (>1.0x)")

        print(f"{p_type:<35} | {avg_audio:<8.2f} | {avg_infer:<8.2f} | {avg_rtf:<6.3f} | {status_str}")

        results["prompts"].append({
            "type": p_type,
            "characters": chars,
            "avg_audio_seconds": round(avg_audio, 2),
            "avg_inference_seconds": round(avg_infer, 3),
            "avg_rtf": round(avg_rtf, 3),
            "chars_per_second": round(char_per_sec, 1),
        })

    print("-" * 70)
    engine.shutdown()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS Engine Benchmark Tool")
    parser.add_argument("--iterations", type=int, default=3, help="Benchmark iterations per prompt")
    parser.add_argument("--voice", default="af_sarah", help="Voice ID")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--output", type=Path, default=None, help="JSON output file path")
    args = parser.parse_args()

    results = run_benchmark(iterations=args.iterations, voice=args.voice, language=args.language)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Benchmark results saved to {args.output}")


if __name__ == "__main__":
    main()
