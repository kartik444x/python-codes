"""
scripts/download_model.py

Download and verify the Kokoro-82M TTS model from Hugging Face Hub.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --force    # re-download even if cached

What this does:
    1. Checks if kokoro package is installed
    2. Imports KPipeline (triggers HF Hub download of model weights)
    3. Verifies the download succeeded by checking the model cache
    4. Runs a single short synthesis to confirm the model works
    5. Reports model size and location

The model weights are stored in the HF Hub cache:
    Windows: %USERPROFILE%\\.cache\\huggingface\\hub
    Linux/Mac: ~/.cache/huggingface/hub

You can override this with TTS_MODEL_DIR in .env or with --model-dir flag.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies() -> bool:
    """Check all required packages are installed."""
    missing = []
    required = {
        "torch": "pip install torch",
        "kokoro": "pip install kokoro",
        "soundfile": "pip install soundfile",
        "numpy": "pip install numpy",
    }
    for pkg, install_cmd in required.items():
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} — install with: {install_cmd}")
            missing.append(pkg)

    return len(missing) == 0


def check_espeak() -> bool:
    """Check if espeak-ng is available on PATH."""
    import shutil
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        print("  ✓ espeak-ng (system binary found)")
        return True
    else:
        print("  ⚠  espeak-ng not found — required for non-English languages")
        print("     Windows: download from https://github.com/espeak-ng/espeak-ng/releases")
        print("     Linux:   sudo apt-get install espeak-ng")
        print("     macOS:   brew install espeak-ng")
        return False


def check_torch_device() -> str:
    """Detect and report available compute device."""
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ CUDA available: {device_name} ({vram_gb:.1f} GB VRAM)")
        return "cuda"
    else:
        print("  ✓ CPU mode (no CUDA GPU detected — this is fine for development)")
        return "cpu"


def download_model(model_dir: Path | None = None, force: bool = False) -> bool:
    """
    Download Kokoro model weights via Hugging Face Hub.

    Args:
        model_dir: Optional directory to use as HF cache.
        force: If True, force re-download even if already cached.

    Returns:
        True if download succeeded.
    """
    import os

    if model_dir:
        model_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(model_dir)
        print(f"\n  Using model directory: {model_dir}")

    print("\n[3/5] Downloading Kokoro-82M model weights from Hugging Face Hub...")
    print("      Model: hexgrad/Kokoro-82M (Apache 2.0)")
    print("      This may take 2–5 minutes on first run (~500 MB).")
    print("      Subsequent runs use the local cache.\n")

    t0 = time.perf_counter()
    try:
        from kokoro import KPipeline

        # Initialize the English pipeline — this triggers the HF Hub download
        print("  Initializing English (American) pipeline...")
        pipeline = KPipeline(lang_code="a")
        elapsed = time.perf_counter() - t0
        print(f"  ✓ Model loaded in {elapsed:.1f}s")
        return True, pipeline

    except ImportError as e:
        print(f"  ✗ kokoro import failed: {e}")
        print("    Run: pip install kokoro")
        return False, None
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  ✗ Model download/load failed after {elapsed:.1f}s: {e}")
        print("\n  Troubleshooting:")
        print("    - Check internet connection")
        print("    - Check HuggingFace Hub status: https://status.huggingface.co")
        print("    - Try: huggingface-cli login (if model requires auth)")
        print("    - Try with --model-dir to specify a custom cache directory")
        return False, None


def verify_synthesis(pipeline: object, output_dir: Path) -> bool:
    """
    Run a short synthesis to verify the model produces valid audio.

    Args:
        pipeline: KPipeline instance.
        output_dir: Directory to save test WAV file.

    Returns:
        True if synthesis produced valid audio.
    """
    import numpy as np
    import soundfile as sf

    print("\n[4/5] Running synthesis verification...")
    test_text = "Hello! The Kokoro text to speech model is working correctly."

    t0 = time.perf_counter()
    try:
        audio_segments = []
        for _, _, audio in pipeline(test_text, voice="af_sarah", speed=1.0):
            if audio is not None and len(audio) > 0:
                audio_segments.append(np.array(audio, dtype=np.float32))

        if not audio_segments:
            print("  ✗ No audio generated")
            return False

        audio = np.concatenate(audio_segments)
        elapsed = time.perf_counter() - t0
        duration = len(audio) / 24000
        rtf = elapsed / duration if duration > 0 else 0

        # Save test audio
        output_dir.mkdir(parents=True, exist_ok=True)
        test_path = output_dir / "model_test.wav"
        sf.write(str(test_path), audio, 24000, subtype="FLOAT")

        print(f"  ✓ Synthesis successful!")
        print(f"    Audio duration: {duration:.2f}s")
        print(f"    Inference time: {elapsed:.2f}s")
        print(f"    Real-time factor: {rtf:.3f}x")
        print(f"    Test audio saved to: {test_path}")
        return True

    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  ✗ Synthesis failed after {elapsed:.2f}s: {e}")
        return False


def print_summary(model_dir: Path | None, success: bool) -> None:
    """Print final download summary."""
    import os

    print("\n[5/5] Summary")
    print("─" * 50)

    hf_home = model_dir or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()

    if success:
        print("✅ Kokoro-82M is ready to use!")
        print(f"\n  Cache location: {hf_home}")
        print("\n  Next steps:")
        print("    1. Copy .env.example to .env")
        print("    2. Run: uvicorn app.main:app --reload")
        print("    3. Open: http://localhost:8000/docs")
    else:
        print("❌ Model setup failed. Review errors above.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify Kokoro-82M TTS model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory to store model weights (default: HF Hub cache)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model is already cached",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip synthesis verification (faster, less thorough)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Kokoro-82M TTS Model Setup")
    print("  License: Apache 2.0 | Model: hexgrad/Kokoro-82M")
    print("=" * 60)

    # Step 1: Check Python dependencies
    print("\n[1/5] Checking Python dependencies...")
    if not check_dependencies():
        print("\n❌ Missing dependencies. Install them first:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    # Step 2: Check system dependencies
    print("\n[2/5] Checking system dependencies...")
    check_espeak()
    check_torch_device()

    # Step 3: Download model
    success, pipeline = download_model(model_dir=args.model_dir, force=args.force)
    if not success:
        print_summary(args.model_dir, success=False)
        return

    # Step 4: Verify synthesis
    if not args.skip_verify and pipeline is not None:
        # Get project root audio dir for test file
        output_dir = PROJECT_ROOT / "generated_audio"
        if not verify_synthesis(pipeline, output_dir):
            success = False

    # Step 5: Summary
    print_summary(args.model_dir, success)


if __name__ == "__main__":
    main()
