"""
KetMaster — AI Mastering Advisor (DeepSeek)
============================================
Takes the rich audio analysis JSON and calls DeepSeek to produce
a complete, genre-aware mastering recipe.

DeepSeek uses the OpenAI-compatible API — same SDK, different base_url.

The recipe controls every parameter of the mastering chain:
  - corrective_eq: list of surgical cuts (freq, db, Q)
  - comp: threshold, ratio, attack_ms, release_ms, makeup_gain_db
  - ms: side_boost_db, low_mono_cutoff_hz
  - tonal_eq: list of additive boosts (freq, db, Q, type)
  - saturation: drive amount (0.0–1.0)
  - target_lufs, true_peak_dbfs: platform targets
  - reasoning: plain English explanation of every decision
  - producer_note: 2–3 sentence note for the producer
"""

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI


# ── Mastering Recipe dataclass ────────────────────────────────────────────────

@dataclass
class EQBand:
    hz:    float
    db:    float
    q:     float  = 1.0
    type:  str    = "peak"   # peak | lowshelf | highshelf | highpass | lowpass
    reason: str   = ""


@dataclass
class CompSettings:
    threshold_db:  float = -20.0
    ratio:         float = 1.5
    attack_ms:     float = 15.0
    release_ms:    float = 300.0
    makeup_gain_db: float = 0.0   # AI can suggest extra makeup if needed
    reason:        str   = ""


@dataclass
class MSSettings:
    side_boost_db:      float = 1.0
    low_mono_cutoff_hz: float = 200.0
    reason:             str   = ""


@dataclass
class MasteringRecipe:
    """Complete mastering instructions from the AI advisor."""
    # Genre + context
    detected_genre:     str   = "unknown"
    detected_mood:      str   = "unknown"
    energy_level:       str   = "medium"     # low | medium | high
    vocal_presence:     str   = "unknown"    # none | low | medium | high
    acoustic_or_electronic: str = "electronic"

    # Chain parameters
    corrective_eq:  list[EQBand]  = field(default_factory=list)
    comp:           CompSettings  = field(default_factory=CompSettings)
    ms:             MSSettings    = field(default_factory=MSSettings)
    tonal_eq:       list[EQBand]  = field(default_factory=list)
    saturation_drive: float       = 0.3     # 0.0 = off, 1.0 = max

    # Targets
    target_lufs:      float = -14.0
    true_peak_dbfs:   float = -1.0

    # Explanation
    reasoning:     str = ""
    producer_note: str = ""

    # Metadata
    model_used: str = ""
    latency_ms: int = 0
    used_fallback: bool = False    # True if AI call failed


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a world-class, complex production, elite mastering engineer with 20+ years of experience across all genres.
You specialise in transforming AI-generated or home‑studio music into professional, commercial and release-ready.

You will receive a complete JSON audio analysis of an audio track. Produce a complete mastering recipe as JSON.
Be SPECIFIC and GENRE-AWARE. A Bollywood track needs different EQ than techno. Never apply the same recipe twice.

VOCAL DETECTION RULES:
- mid_ratio > 0.25 AND spectral_centroid_hz 1500-4000 Hz -> set vocal_presence to "medium" or "high"
- harmonic_ratio > 0.6 AND percussive_ratio < 0.3 -> likely vocal/acoustic, set acoustic_or_electronic to "acoustic" or "hybrid"
- AI vocals ALWAYS have harshness at 3.2 kHz and synthetic sibilance at 8 kHz
  -> include corrective_eq cut at 3200 Hz if upper_mid_ratio > 0.12
  -> include corrective_eq cut at 8000 Hz if air_ratio > 0.09

COMPRESSOR RULES:
- ratio: 1.3-2.0. Never above 2.5. BPM-lock release: half-beat = 60000/bpm*0.5 ms
- attack: 10-30ms percussive, 20-60ms harmonic/vocal. Always loudness-neutral (makeup_gain_db compensates).

SATURATION MODEL SELECTION (acoustic_or_electronic field drives tape/tube/clip blend):
- acoustic/jazz/classical: saturation_drive 0.10-0.25, acoustic_or_electronic="acoustic"
- pop/R&B/soul: saturation_drive 0.25-0.40, acoustic_or_electronic="hybrid"
- rock/metal: saturation_drive 0.35-0.55, acoustic_or_electronic="hybrid"
- electronic/EDM/trap: saturation_drive 0.45-0.70, acoustic_or_electronic="electronic"

M/S RULES:
- stereo_correlation > 0.92 = too narrow -> side_boost_db: +1.5 to +2.5
- stereo_correlation < 0.5 = phase issues -> side_boost_db: -1.5 to -3.0
- Always mono below 150-200 Hz

TONAL EQ (additive, max 4 bands): low shelf 80-120 Hz warmth, presence peak 3-5 kHz, high shelf 10-14 kHz air.
CORRECTIVE EQ (cuts only, max 4 bands): high-pass at 20-40 Hz always. Cut resonances from iso_band_energies.

Rules:
- corrective_eq: only cut problem frequencies. Max 4 bands. Cuts only (negative db).
- comp: BPM-locked release. ratio 1.3-2.0. Never above 2.5. Always loudness-neutral.
- ms.side_boost_db: 0.0-3.0 for narrow, -1.0 to -3.0 for phase issues.
- tonal_eq: add character. Max 4 bands. Shelves for overall tone, peaks for character.
- Always explain WHY each decision was made in reasoning.
- producer_note: max 3 sentences, plain English, honest about what was fixed.
- If noise_floor_db > -40 dBFS:
  - Add a second high‑shelf cut at 10 kHz, -3 to -5 dB.
  - Consider a downward expander or additional gate.
  - Mention in producer_note that noise was noticeable and has been reduced as much as possible without damaging the high‑end.

Respond ONLY with valid JSON matching this exact schema (no markdown, no preamble):

{
  "detected_genre": "string",
  "detected_mood": "string",
  "energy_level": "low|medium|high",
  "vocal_presence": "none|low|medium|high",
  "acoustic_or_electronic": "acoustic|electronic|hybrid",
  "corrective_eq": [
    {"hz": 0, "db": 0, "q": 1.0, "type": "peak|highpass|lowpass", "reason": "string"}
  ],
  "comp": {
    "threshold_db": -20, "ratio": 1.5, "attack_ms": 15,
    "release_ms": 300, "makeup_gain_db": 0, "reason": "string"
  },
  "ms": {
    "side_boost_db": 1.0, "low_mono_cutoff_hz": 200, "reason": "string"
  },
  "tonal_eq": [
    {"hz": 0, "db": 0, "q": 1.0, "type": "peak|lowshelf|highshelf", "reason": "string"}
  ],
  "saturation_drive": 0.3,
  "target_lufs": -14.0,
  "true_peak_dbfs": -1.0,
  "reasoning": "Full paragraph explaining all decisions for THIS specific track.",
  "producer_note": "2-3 sentence honest note about what was fixed and any remaining issues."
}
"""


# ── Main advisor function ─────────────────────────────────────────────────────

def get_mastering_recipe(
    analysis_json: str,
    platform: str = "spotify",
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    timeout_seconds: int = 30,
) -> MasteringRecipe:
    """
    Call DeepSeek with the analysis JSON and return a MasteringRecipe.
    Falls back to a sensible default recipe if the API call fails.

    Args:
        analysis_json:   Output of audio_analyser.to_json()
        platform:        Target platform name (used as context hint)
        api_key:         DeepSeek API key (or set DEEPSEEK_API_KEY env var)
        model:           "deepseek-chat" (default) or "deepseek-reasoner"
        timeout_seconds: Max seconds to wait for API response
    """
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")

    if not key:
        print("  [AI Advisor] No API key — using rule-based fallback")
        return _rule_based_fallback(analysis_json, platform)

    client = OpenAI(
        api_key=key,
        base_url="https://api.deepseek.com",
    )

        # Parse analysis_json to extract noise floor
    try:
        analysis_dict = json.loads(analysis_json)
        noise_floor = analysis_dict.get('noise_floor_db', -99)
    except:
        noise_floor = -99

    user_message = f"""Analyse this track and produce a mastering recipe.
Target platform: {platform}

Track analysis:
{analysis_json}

Special note: The noise floor is {noise_floor} dBFS.
If it's above -50 dBFS, you may suggest a noise gate or a high‑shelf cut above 12 kHz.
"""

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.2,   # low temp = consistent, repeatable decisions
            max_tokens=1500,
            timeout=timeout_seconds,
        )
        raw = response.choices[0].message.content.strip()
        latency = int((time.time() - t0) * 1000)

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        recipe = _parse_recipe(raw)
        recipe.model_used   = model
        recipe.latency_ms   = latency
        recipe.used_fallback = False
        return recipe

    except Exception as e:
        print(f"  [AI Advisor] API error ({e}) — using rule-based fallback")
        recipe = _rule_based_fallback(analysis_json, platform)
        recipe.used_fallback = True
        return recipe


def _parse_recipe(raw_json: str) -> MasteringRecipe:
    """Parse the AI JSON response into a MasteringRecipe dataclass."""
    data = json.loads(raw_json)
    recipe = MasteringRecipe()

    recipe.detected_genre          = data.get("detected_genre", "unknown")
    recipe.detected_mood           = data.get("detected_mood", "unknown")
    recipe.energy_level            = data.get("energy_level", "medium")
    recipe.vocal_presence          = data.get("vocal_presence", "medium")
    recipe.acoustic_or_electronic  = data.get("acoustic_or_electronic", "electronic")
    recipe.saturation_drive        = float(data.get("saturation_drive", 0.3))
    recipe.target_lufs             = float(data.get("target_lufs", -14.0))
    recipe.true_peak_dbfs          = float(data.get("true_peak_dbfs", -1.0))
    recipe.reasoning               = data.get("reasoning", "")
    recipe.producer_note           = data.get("producer_note", "")

    # Corrective EQ
    for b in data.get("corrective_eq", []):
        recipe.corrective_eq.append(EQBand(
            hz=float(b.get("hz", 0)),
            db=float(b.get("db", 0)),
            q=float(b.get("q", 1.0)),
            type=b.get("type", "peak"),
            reason=b.get("reason", ""),
        ))

    # Compressor
    c = data.get("comp", {})
    recipe.comp = CompSettings(
        threshold_db  = float(c.get("threshold_db", -20)),
        ratio         = float(c.get("ratio", 1.5)),
        attack_ms     = float(c.get("attack_ms", 15)),
        release_ms    = float(c.get("release_ms", 300)),
        makeup_gain_db = float(c.get("makeup_gain_db", 0)),
        reason        = c.get("reason", ""),
    )

    # M/S
    m = data.get("ms", {})
    recipe.ms = MSSettings(
        side_boost_db      = float(m.get("side_boost_db", 1.0)),
        low_mono_cutoff_hz = float(m.get("low_mono_cutoff_hz", 200)),
        reason             = m.get("reason", ""),
    )

    # Tonal EQ
    for b in data.get("tonal_eq", []):
        recipe.tonal_eq.append(EQBand(
            hz=float(b.get("hz", 0)),
            db=float(b.get("db", 0)),
            q=float(b.get("q", 0.7)),
            type=b.get("type", "peak"),
            reason=b.get("reason", ""),
        ))

    return recipe


# ── Rule-based fallback (no API key needed) ───────────────────────────────────

def _rule_based_fallback(analysis_json: str, platform: str) -> MasteringRecipe:
    """
    Produce a decent recipe from rules when the AI is unavailable.
    Uses the analysis data directly — much smarter than hardcoded values.
    """
    try:
        data = json.loads(analysis_json)
    except Exception:
        data = {}

    recipe = MasteringRecipe()
    recipe.model_used    = "rule-based-fallback"
    recipe.used_fallback = True

    bpm          = float(data.get("detected_bpm", 120))
    low_ratio    = float(data.get("bass_ratio", 0) + data.get("sub_bass_ratio", 0))
    upper_ratio  = float(data.get("upper_mid_ratio", 0))
    air_ratio    = float(data.get("air_ratio", 0))
    centroid     = float(data.get("spectral_centroid_hz", 2000))
    corr         = float(data.get("stereo_correlation", 0.9))
    dr           = float(data.get("dynamic_range_db", 10))
    perc_ratio   = float(data.get("percussive_ratio", 0.5))

    # Corrective EQ
    recipe.corrective_eq.append(EQBand(hz=30, db=0, q=0.7, type="highpass", reason="remove sub-rumble"))
    if low_ratio > 0.35:
        recipe.corrective_eq.append(EQBand(hz=280, db=-2.0, q=0.9, type="peak", reason=f"low-end {low_ratio:.2f} too dominant"))
    if centroid < 1800:
        recipe.corrective_eq.append(EQBand(hz=400, db=-1.5, q=1.2, type="peak", reason="boxiness"))
    if upper_ratio > 0.15:
        recipe.corrective_eq.append(EQBand(hz=3200, db=-1.5, q=1.5, type="peak", reason="digital harshness"))

    # Compressor — BPM-locked release
    half_beat_ms = (60000 / max(60, bpm)) * 0.5
    recipe.comp = CompSettings(
        threshold_db  = -20.0,
        ratio         = 1.8 if perc_ratio > 0.6 else 1.4,
        attack_ms     = 10.0 if perc_ratio > 0.6 else 20.0,
        release_ms    = round(half_beat_ms, 1),
        makeup_gain_db = 0.0,
        reason        = f"BPM-locked release at {half_beat_ms:.0f}ms (half-beat at {bpm:.0f} BPM)",
    )

    # M/S
    if corr > 0.92:
        side_boost = 2.0
    elif corr < 0.5:
        side_boost = -2.0
    else:
        side_boost = 1.0
    recipe.ms = MSSettings(side_boost_db=side_boost, low_mono_cutoff_hz=200.0, reason="stereo width correction")

    # Tonal EQ
    warmth = 1.5 if low_ratio < 0.25 else 0.5
    recipe.tonal_eq.append(EQBand(hz=100,   db=warmth, q=0.7, type="lowshelf",  reason="warmth"))
    recipe.tonal_eq.append(EQBand(hz=5000,  db=0.8,   q=1.4, type="peak",      reason="presence"))
    air = 2.0 if air_ratio < 0.08 else 1.0
    recipe.tonal_eq.append(EQBand(hz=14000, db=air,   q=0.7, type="highshelf", reason="air / sheen"))

    recipe.target_lufs    = {"spotify":-14,"apple_music":-16,"youtube":-14,
                             "soundcloud":-11,"tidal":-14,"beatport":-8}.get(platform, -14.0)
    recipe.true_peak_dbfs = -1.0
    recipe.reasoning      = "Rule-based fallback: parameters derived from signal analysis without AI."
    recipe.producer_note  = "Processed with rule-based settings. Add a DeepSeek API key for AI-driven decisions."

    return recipe