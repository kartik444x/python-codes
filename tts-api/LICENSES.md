# Licensing, Legal & Ethical Governance Specification

Treating licensing, privacy, and voice safety as first-class architectural concerns.

---

## 1. Primary Codebase License

This software application (`tts-api`) is licensed under the **MIT License**.

```
MIT License
Copyright (c) 2026 TTS API Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 2. TTS ML Model License

| Model Component | Provider | License | Commercial Use | Redistribution Terms |
| :--- | :--- | :--- | :--- | :--- |
| **Kokoro-82M Weights** | `hexgrad/Kokoro-82M` | **Apache 2.0** | ✅ Permitted | Notice and license attribution |
| **Misaki G2P** | `hexgrad/misaki` | **Apache 2.0** | ✅ Permitted | Notice and license attribution |
| **PyTorch** | Meta AI | **BSD-3-Clause** | ✅ Permitted | BSD copyright notice |
| **Hugging Face Hub** | Hugging Face | **Apache 2.0** | ✅ Permitted | Standard Apache notice |

### Why We Selected Kokoro-82M Over Alternatives
- **XTTS-v2**: Uses **Coqui Public Model License (CPML)**, which prohibits commercial competition and holds non-standard restrictions. Coqui company ceased operations in 2024.
- **SpeechT5**: Uses speaker embeddings derived from non-commercial or academic research corpora.
- **Kokoro-82M**: Full **Apache 2.0** open-weight license permitting commercial self-hosting without recurring character royalties.

---

## 3. System Dependencies

| Dependency | Purpose | License | Production Compliance Strategy |
| :--- | :--- | :--- | :--- |
| **espeak-ng** | Multilingual G2P phonemizer | **GPL-3.0** | Invoked as an external dynamic subprocess / system CLI binary. Not statically linked or embedded into proprietary application bytecode. |
| **ffmpeg** | WAV to MP3 transcoding | **LGPL-2.1+ / GPL** | Invoked via subprocess by `pydub`. Complies with LGPL dynamic runtime requirements. |
| **libsndfile** | High-precision audio I/O | **LGPL-2.1+** | Loaded dynamically via CFFI / `soundfile`. |

---

## 4. Voice Dataset & Cloning Safeguards

1. **Preset Voice Profiles:**
   All 24 preset voices (`af_sarah`, `am_adam`, `bf_emma`, etc.) bundled with Kokoro-82M are derived from legally permitted open speech training corpora.
2. **Anti-Impersonation Safeguards:**
   - The platform prohibits synthesizing speech intended to deceptively impersonate living persons or public figures without written authorization.
   - The `POST /api/v1/voices` registration API strictly validates owner consent flags (`consent_confirmed=true`) and stores audit trail records.
