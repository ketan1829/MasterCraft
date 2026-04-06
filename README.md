# MasterCraft - AI‑Driven Audio Mastering

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DeepSeek](https://img.shields.io/badge/AI-DeepSeek-4A90E2)](https://deepseek.com)

**MasterCraft** is a professional, AI‑powered audio mastering tool that analyses your track and applies a custom mastering chain – just like a real mastering engineer. It uses **DeepSeek** (via OpenAI‑compatible API) to understand genre, dynamics, frequency balance, and stereo image, then generates a precise recipe for corrective EQ, compression, M/S processing, tonal EQ, saturation, and loudness normalisation.

No more presets. No more guesswork. Every decision is data‑driven and tailored to your song.

---

## Features

- **Rich audio analysis** – LUFS, true peak, LRA, dynamic range, 6‑zone frequency balance, 31‑band ISO spectrum, BPM, key, stereo correlation, noise floor, and more.
- **AI mastering advisor** – DeepSeek analyses the track and outputs a complete, genre‑aware mastering recipe (EQ cuts, compression, M/S width, saturation, etc.).
- **Full mastering chain** – DC offset fix, corrective EQ, glue compression, M/S width, tonal EQ, harmonic saturation, LUFS normalisation, true‑peak limiting.
- **Platform‑specific targets** – Spotify, Apple Music, YouTube, SoundCloud, Tidal, Beatport, or custom LUFS/TP.
- **Detailed report** – Before/after metrics, frequency balance chart, AI reasoning, producer notes, and QA status.
- **Rule‑based fallback** – Works without an API key (sensible defaults derived from analysis).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ketan1829/MasterCraft.git
cd MasterCraft

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


🤝 Contributing
Issues and pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

🙏 Acknowledgements
- DeepSeek for the affordable, high‑quality AI.
- librosa for audio analysis.
- pedalboard by Spotify for the DSP building blocks.
- pyloudnorm for ITU‑R BS.1770‑4 loudness measurement.

💬 Questions?
Open an issue on GitHub

Master smarter, not harder. 🎧
