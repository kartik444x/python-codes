"""
scripts/warmup.py

Model warm-up verification script.
Loads the TTS engine and runs timed warm-up inferences.

Use this to:
- Verify the model loads correctly before starting the API server
- Measure warm-start inference times on your hardware
- Confirm that model caching is working

Usage:
    python scripts/warmup.py
    python scripts/warmup.py --iterations 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WARMUP_TEXTS = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "This is a production text to speech API powered by Kokoro.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS model warm-up verification")
    parser.add_argument("--iterations", type=int, default=3, help="Number of warm-up runs")
    parser.add_argument("--voice", default="af_sarah", help="Voice ID to use")
    parser.add_argument("--language", default="en", help="Language code")
    args = parser.parse_args()

    print("=" * 60)
    print("  TTS Engine Warm-up Verification")
    print("=" * 60)

    # Load settings + engine
    from app.core.config import get_settings
    from app.engines import create_engine

    settings = get_settings()
    print(f"\n  Device:  {settings.resolved_tts_device}")
    print(f"  Model:   {settings.tts_model.value}")
    print(f"  Voice:   {args.voice}")

    # Load engine
    print("\n[1/3] Loading model...")
    t0 = time.perf_counter()
    engine = create_engine()
    engine.load()
    load_time = time.perf_counter() - t0
    print(f"  ✓ Model loaded in {load_time:.2f}s")

    # Warm-up iterations
    print(f"\n[2/3] Running {args.iterations} warm-up iteration(s)...")
    durations = []
    rtfs = []

    for i in range(args.iterations):
        text = WARMUP_TEXTS[i % len(WARMUP_TEXTS)]
        t0 = time.perf_counter()
        result = engine.synthesize(
            text=text,
            voice_id=args.voice,
            language=args.language,
            speed=1.0,
        )
        elapsed = time.perf_counter() - t0
        rtf = elapsed / result.duration_seconds if result.duration_seconds > 0 else 0
        durations.append(elapsed)
        rtfs.append(rtf)
        print(f"  Run {i+1}: {elapsed*1000:.0f}ms inference | {result.duration_seconds:.2f}s audio | RTF={rtf:.3f}")

    # Summary
    print(f"\n[3/3] Results")
    print("─" * 40)
    print(f"  Average inference: {sum(durations)/len(durations)*1000:.0f}ms")
    print(f"  Average RTF:       {sum(rtfs)/len(rtfs):.3f}")
    print(f"  Best RTF:          {min(rtfs):.3f}")
    print(f"\n  RTF < 1.0 = faster than real-time ✓")

    if min(rtfs) < 1.0:
        print("\n✅ Model is performing well and is ready for production.")
    else:
        print("\n⚠  RTF > 1.0: inference is slower than real-time.")
        print("   Consider enabling GPU: TTS_DEVICE=cuda in .env")

    engine.shutdown()


if __name__ == "__main__":
    main()
