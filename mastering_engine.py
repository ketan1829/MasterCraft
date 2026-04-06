"""
KetMaster — Professional Mastering Engine v2
=============================================
AI-driven mastering chain. Every parameter is set by DeepSeek based
on rich audio analysis — not hardcoded rules.

Architecture:
  audio_analyser.py  →  RichAudioAnalysis (30+ descriptors)
  ai_advisor.py      →  MasteringRecipe   (AI decisions)
  mastering_engine.py→  Applies recipe, writes master WAV

Chain:
  1.  Load + resample to 44100 Hz / 32-bit float
  2.  Rich analysis   (30+ features)
  3.  AI advisor      (DeepSeek → mastering recipe)
  4.  DC offset fix
  5.  Corrective EQ   (AI-specified cuts)
  6.  Glue comp       (BPM-locked, loudness-neutral)
  7.  M/S processing  (AI-specified width + mono bass)
  8.  Tonal EQ        (AI-specified additive shaping)
  9.  Harmonic saturation (AI-specified drive)
  10. Loudness normalisation + true-peak ceiling
  11. Post-chain LUFS verification trim
  12. Write 24-bit WAV
"""

from __future__ import annotations
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import librosa
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
import pedalboard as pb
import json
import time

from audio_analyser import analyse_track, to_json, RichAudioAnalysis
from ai_advisor import get_mastering_recipe, MasteringRecipe, EQBand

warnings.filterwarnings("ignore")

# ── Platform targets (AI can override) ────────────────────────────────────────
PLATFORM_TARGETS = {
    "spotify":      {"lufs": -14.0, "true_peak": -1.0},
    "apple_music":  {"lufs": -16.0, "true_peak": -1.0},
    "youtube":      {"lufs": -14.0, "true_peak": -1.0},
    "soundcloud":   {"lufs": -11.0, "true_peak": -0.5},
    "tidal":        {"lufs": -14.0, "true_peak": -1.0},
    "amazon_music": {"lufs": -14.0, "true_peak": -1.0},
    "beatport":     {"lufs": -8.0,  "true_peak": -0.3},
    "producer_mix": {"lufs": -6.0,  "true_peak": -0.1},
}


# ── Report dataclass ──────────────────────────────────────────────────────────
@dataclass
class MasteringReport:
    input_path:       str = ""
    output_path:      str = ""
    platform:         str = "spotify"
    target_lufs:      float = -14.0
    target_true_peak: float = -1.0

    # Analysis snapshots
    before: RichAudioAnalysis = field(default_factory=RichAudioAnalysis)
    after:  RichAudioAnalysis = field(default_factory=RichAudioAnalysis)

    # AI recipe (what the AI decided)
    recipe: Optional[MasteringRecipe] = None

    # Stage gain log
    corrective_eq_applied: list[dict] = field(default_factory=list)
    glue_comp_gr_db:       float = 0.0
    ms_side_db:            float = 0.0
    tonal_eq_applied:      list[dict] = field(default_factory=list)
    saturation_drive:      float = 0.0
    limiter_gain_db:       float = 0.0

    warnings:   list[str] = field(default_factory=list)
    passed_qa:  bool = False
    elapsed_s:  float = 0.0


# ── Helper: build pedalboard from EQBand list ─────────────────────────────────
def _build_eq_board(bands: list[EQBand]) -> Optional[pb.Pedalboard]:
    plugins = []
    for b in bands:
        hz = max(10.0, min(20000.0, float(b.hz)))
        db = float(b.db)
        q  = float(b.q) if b.q else 1.0
        t  = b.type.lower()
        if t == "highpass":
            plugins.append(pb.HighpassFilter(cutoff_frequency_hz=hz))
        elif t == "lowpass":
            plugins.append(pb.LowpassFilter(cutoff_frequency_hz=hz))
        elif t == "lowshelf":
            plugins.append(pb.LowShelfFilter(cutoff_frequency_hz=hz, gain_db=db, q=q))
        elif t == "highshelf":
            plugins.append(pb.HighShelfFilter(cutoff_frequency_hz=hz, gain_db=db, q=q))
        else:  # peak (default)
            plugins.append(pb.PeakFilter(cutoff_frequency_hz=hz, gain_db=db, q=q))
    return pb.Pedalboard(plugins) if plugins else None


# ── Stage implementations ──────────────────────────────────────────────────────

def _stage_dc_fix(audio: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    sos = butter(2, 5, btype='high', fs=sr, output='sos')
    return sosfilt(sos, audio, axis=0).astype(np.float32)


def _stage_corrective_eq(
    audio: np.ndarray, sr: int,
    recipe: MasteringRecipe, report: MasteringReport
) -> np.ndarray:
    bands = recipe.corrective_eq
    if not bands:
        return audio
    board = _build_eq_board(bands)
    if not board:
        return audio
    processed = board(audio.T.astype(np.float32), sr).T
    report.corrective_eq_applied = [
        {"hz": b.hz, "db": b.db, "q": b.q, "type": b.type, "reason": b.reason}
        for b in bands
    ]
    return processed


def _stage_glue_compression(
    audio: np.ndarray, sr: int,
    recipe: MasteringRecipe, report: MasteringReport
) -> np.ndarray:
    meter = pyln.Meter(sr)
    lufs_before = float(meter.integrated_loudness(audio))

    comp = recipe.comp
    board = pb.Pedalboard([
        pb.Compressor(
            threshold_db = float(comp.threshold_db),
            ratio        = float(np.clip(comp.ratio, 1.0, 4.0)),
            attack_ms    = float(np.clip(comp.attack_ms, 1.0, 100.0)),
            release_ms   = float(np.clip(comp.release_ms, 50.0, 1000.0)),
        )
    ])
    result = board(audio.T.astype(np.float32), sr).T

    lufs_after = float(meter.integrated_loudness(result))
    gr_db = lufs_before - lufs_after
    report.glue_comp_gr_db = round(gr_db, 2)

    # Makeup gain: AI can request extra; we always at least restore what was lost
    total_makeup = gr_db + float(comp.makeup_gain_db)
    if total_makeup > 0.1:
        result = (result * 10 ** (total_makeup / 20.0)).astype(np.float32)

    return result


def _stage_ms_processing(
    audio: np.ndarray, sr: int,
    recipe: MasteringRecipe, report: MasteringReport
) -> np.ndarray:
    if audio.ndim < 2 or audio.shape[1] != 2:
        return audio

    L, R = audio[:, 0].copy(), audio[:, 1].copy()
    M = (L + R) * 0.5
    S = (L - R) * 0.5

    # Mono-ise low end below AI-specified cutoff
    from scipy.signal import butter, sosfiltfilt
    cutoff = float(np.clip(recipe.ms.low_mono_cutoff_hz, 60, 400))
    sos_hp = butter(4, cutoff, btype='high', fs=sr, output='sos')
    S_high = sosfiltfilt(sos_hp, S).astype(np.float32)

    side_boost = float(np.clip(recipe.ms.side_boost_db, -6.0, 6.0))
    S_final = S_high * 10 ** (side_boost / 20.0)

    report.ms_side_db = side_boost

    L_new = (M + S_final).astype(np.float32)
    R_new = (M - S_final).astype(np.float32)
    return np.stack([L_new, R_new], axis=1)


def _stage_tonal_eq(
    audio: np.ndarray, sr: int,
    recipe: MasteringRecipe, report: MasteringReport
) -> np.ndarray:
    bands = recipe.tonal_eq
    if not bands:
        return audio
    board = _build_eq_board(bands)
    if not board:
        return audio
    processed = board(audio.T.astype(np.float32), sr).T
    report.tonal_eq_applied = [
        {"hz": b.hz, "db": b.db, "q": b.q, "type": b.type, "reason": b.reason}
        for b in bands
    ]
    return processed


def _stage_saturation(
    audio: np.ndarray, recipe: MasteringRecipe, report: MasteringReport
) -> np.ndarray:
    drive = float(np.clip(recipe.saturation_drive, 0.0, 1.0))
    report.saturation_drive = drive
    if drive < 0.05:
        return audio

    # Soft-clip threshold scales with drive: 0.0→no clip, 1.0→0.5 (-6dBFS threshold)
    threshold = 1.0 - (drive * 0.5)    # range: 1.0 (no clip) → 0.5 (-6dBFS)
    knee_width = 1.0 - threshold        # how far above threshold tanh kicks in

    audio_out = np.where(
        np.abs(audio) > threshold,
        np.sign(audio) * (threshold + knee_width * np.tanh(
            (np.abs(audio) - threshold) / (knee_width + 1e-6)
        )),
        audio
    ).astype(np.float32)
    return audio_out


def _stage_normalize(
    audio: np.ndarray, sr: int,
    target_lufs: float, target_tp: float,
    report: MasteringReport
) -> np.ndarray:
    """Two-pass LUFS normalization + true-peak ceiling."""
    meter = pyln.Meter(sr)
    ceiling_linear = 10 ** (target_tp / 20.0)

    # Pass 1 — LUFS gain
    current_lufs = float(meter.integrated_loudness(audio))
    if not np.isfinite(current_lufs) or current_lufs < -70:
        current_lufs = -23.0
    gain_db = float(np.clip(target_lufs - current_lufs, -30.0, 30.0))
    audio = (audio * 10 ** (gain_db / 20.0)).astype(np.float32)

    # Pass 2 — true-peak ceiling (oversample loudest 2s window)
    window_samples = sr * 2
    mono_c = audio.mean(axis=1)
    if len(mono_c) > window_samples:
        rms_windows = np.array([
            np.sqrt(np.mean(mono_c[i:i+window_samples]**2))
            for i in range(0, len(mono_c)-window_samples, window_samples//4)
        ])
        start = np.argmax(rms_windows) * (window_samples // 4)
        peak_window = audio[start:start+window_samples]
    else:
        peak_window = audio

    from scipy.signal import resample_poly
    window_4x = resample_poly(peak_window, up=4, down=1, axis=0).astype(np.float32)
    # Pass 2 — true-peak limiter (lookahead brickwall)
    from pedalboard import Limiter

    # Get current peak after LUFS gain
    current_peak = float(np.abs(audio).max())
    ceiling_linear = 10 ** (target_tp / 20.0)

    # Apply limiter only if peaks exceed ceiling
    if current_peak > ceiling_linear:
        limiter = Limiter(threshold_db=target_tp, release_ms=10)
        # Limiter expects shape (channels, samples) -> pedalboard v0.9+
        audio = limiter(audio.T, sr).T
        reduction_db = float(20 * np.log10(current_peak / (np.abs(audio).max() + 1e-10)))
    else:
        reduction_db = 0.0

    total_gain_db = gain_db - reduction_db   # limiter reduces gain only on peaks
    report.limiter_gain_db = round(total_gain_db, 2)
    return audio


def _post_chain_trim(
    audio: np.ndarray,
    sr: int,
    target_lufs: float,
    target_tp: float,
    report: MasteringReport
) -> np.ndarray:
    """Final verification: re-measure LUFS and apply a clean linear trim if needed."""
    meter = pyln.Meter(sr)
    final_lufs = float(meter.integrated_loudness(audio))
    ceiling_linear = 10 ** (target_tp / 20.0)

    if np.isfinite(final_lufs) and final_lufs > -70:
        trim_db = target_lufs - final_lufs
        # Allow up to 1.5 dB trim (was 0.1)
        if abs(trim_db) > 0.3:
            candidate = (audio * 10 ** (trim_db / 20.0)).astype(np.float32)
            # Check candidate peaks; if they exceed, re‑limit instead of clipping
            if float(np.abs(candidate).max()) <= ceiling_linear * 1.02:
                # Safe to apply linear gain
                audio = candidate
                report.limiter_gain_db = round(report.limiter_gain_db + trim_db, 2)
            else:
                # Re‑apply limiter with adjusted threshold
                from pedalboard import Limiter
                # Estimate new threshold to accommodate extra gain
                peak_after = float(np.abs(candidate).max())
                if peak_after > 1e-6:
                    new_thresh = target_tp - float(20 * np.log10(peak_after / ceiling_linear))
                    limiter = Limiter(threshold_db=new_thresh, release_ms=10)
                    audio = limiter(candidate.T, sr).T
                    report.limiter_gain_db = round(report.limiter_gain_db + trim_db, 2)

    # Final safety clip (should be unnecessary but safe)
    return np.clip(audio, -ceiling_linear, ceiling_linear).astype(np.float32)
# ── Main function ─────────────────────────────────────────────────────────────

def master_track(
    input_path:        str | Path,
    output_path:       str | Path,
    platform:          str = "spotify",
    api_key:           Optional[str] = None,
    custom_lufs:       Optional[float] = None,
    custom_true_peak:  Optional[float] = None,
    force_fallback:    bool = False,   # skip AI, use rule-based
) -> MasteringReport:
    """
    Run the full AI-driven mastering chain.

    Args:
        input_path:       Any audio file (WAV, MP3, FLAC, AIFF, OGG)
        output_path:      Output path — always written as 24-bit WAV
        platform:         Target platform key or 'custom'
        api_key:          DeepSeek API key (or set DEEPSEEK_API_KEY env var)
        custom_lufs:      Override target LUFS
        custom_true_peak: Override true-peak ceiling
        force_fallback:   Skip AI call entirely (useful for testing)
    """
    t0 = time.time()
    input_path  = Path(input_path)
    output_path = Path(output_path)

    # ── Resolve platform targets ────────────────────────────────────────────
    if platform in PLATFORM_TARGETS:
        target_lufs = custom_lufs  or PLATFORM_TARGETS[platform]["lufs"]
        target_tp   = custom_true_peak or PLATFORM_TARGETS[platform]["true_peak"]
    elif custom_lufs and custom_true_peak:
        target_lufs = custom_lufs
        target_tp   = custom_true_peak
    else:
        raise ValueError(f"Unknown platform '{platform}'. Options: {list(PLATFORM_TARGETS)}")

    report = MasteringReport(
        input_path=str(input_path),
        output_path=str(output_path),
        platform=platform,
        target_lufs=target_lufs,
        target_true_peak=target_tp,
    )

    # ── Load audio ──────────────────────────────────────────────────────────
    audio, sr = sf.read(str(input_path), dtype="float32", always_2d=True)

    # Ensure stereo
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    elif audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)

    # Resample to 44100 if needed
    if sr != 44100:
        L = librosa.resample(audio[:, 0], orig_sr=sr, target_sr=44100)
        R = librosa.resample(audio[:, 1], orig_sr=sr, target_sr=44100)
        audio = np.stack([L, R], axis=1).astype(np.float32)
        sr = 44100

    # ── Stage 1: Rich analysis ──────────────────────────────────────────────
    print("  [1/10] Analysing track...", flush=True)
    analysis = analyse_track(audio, sr, str(input_path))
    report.before = analysis
    analysis_json = to_json(analysis)

    # QA pre-warnings
    if analysis.stereo_correlation < 0.3:
        report.warnings.append("⚠ Severe phase issues — check mono compatibility")
    if analysis.dynamic_range_db < 5:
        report.warnings.append("⚠ Very low dynamic range — may already be over-compressed")
    if analysis.noise_floor_db > -50:
        report.warnings.append(f"⚠ High noise floor at {analysis.noise_floor_db:.1f} dBFS")
    if analysis.low_end_mono < 0.7:
        report.warnings.append("⚠ Bass is not mono — may cause phase issues on mono playback")

    # ── Stage 2: AI mastering recipe ────────────────────────────────────────
    print("  [2/10] Consulting AI advisor...", flush=True)
    if force_fallback:
        from ai_advisor import _rule_based_fallback
        recipe = _rule_based_fallback(analysis_json, platform)
    else:
        recipe = get_mastering_recipe(
            analysis_json, platform=platform, api_key=api_key
        )
    report.recipe = recipe

    # AI can override targets — let it, but clamp to safety
    target_lufs = float(np.clip(recipe.target_lufs, -20.0, -6.0))
    target_tp   = float(np.clip(recipe.true_peak_dbfs, -3.0, -0.1))
    report.target_lufs      = target_lufs
    report.target_true_peak = target_tp

    # ── Stage 3: DC fix ─────────────────────────────────────────────────────
    print("  [3/10] DC offset fix...", flush=True)
    audio = _stage_dc_fix(audio, sr)

    # ── Stage 3.5: Noise gate (if needed) ──────────────────────────────────
    if analysis.noise_floor_db > -50:   # -50 dBFS threshold
        print("  [3.5/10] Noise gate (floor > -50 dBFS)...", flush=True)
        # from pedalboard import Gate
        from pedalboard import NoiseGate

        # Threshold set 6 dB above noise floor
        gate_thresh = analysis.noise_floor_db + 6.0
        # gate = Gate(threshold_db=gate_thresh, attack_ms=5, release_ms=100, ratio=4.0)
        gate = NoiseGate(threshold_db=gate_thresh, attack_ms=5, release_ms=100)

        audio = gate(audio.T, sr).T
        report.warnings.append(f"⚠ Applied noise gate (threshold {gate_thresh:.1f} dBFS)")

    # ── Stage 4: Corrective EQ ──────────────────────────────────────────────
    print(f"  [4/10] Corrective EQ ({len(recipe.corrective_eq)} bands)...", flush=True)
    audio = _stage_corrective_eq(audio, sr, recipe, report)

    # ── Stage 5: Glue compression ───────────────────────────────────────────
    print(f"  [5/10] Glue compression ({recipe.comp.ratio:.1f}:1 @ {recipe.comp.threshold_db:.0f}dB)...", flush=True)
    audio = _stage_glue_compression(audio, sr, recipe, report)

    # ── Stage 6: M/S processing ─────────────────────────────────────────────
    print(f"  [6/10] M/S processing (side {recipe.ms.side_boost_db:+.1f}dB)...", flush=True)
    audio = _stage_ms_processing(audio, sr, recipe, report)

    # ── Stage 7: Tonal EQ ───────────────────────────────────────────────────
    print(f"  [7/10] Tonal EQ ({len(recipe.tonal_eq)} bands)...", flush=True)
    audio = _stage_tonal_eq(audio, sr, recipe, report)

    # ── Stage 8: Harmonic saturation ────────────────────────────────────────
    print(f"  [8/10] Saturation (drive={recipe.saturation_drive:.2f})...", flush=True)
    audio = _stage_saturation(audio, recipe, report)

    # ── Stage 9: Normalise + true-peak ceiling ──────────────────────────────
    print(f"  [9/10] Normalising to {target_lufs:.0f} LUFS / {target_tp:.1f} dBTP...", flush=True)
    audio = _stage_normalize(audio, sr, target_lufs, target_tp, report)

    # ── Stage 10: Post-chain verification trim ──────────────────────────────
    print("  [10/10] Verification trim...", flush=True)
    audio = _post_chain_trim(audio, sr, target_lufs, target_tp, report)

    # ── Write 24-bit WAV ────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr, subtype="PCM_24")

    # ── Post analysis + QA ──────────────────────────────────────────────────
    report.after = analyse_track(audio, sr, str(output_path))
    report.elapsed_s = round(time.time() - t0, 1)

    lufs_ok = abs(report.after.integrated_lufs - target_lufs) < 1.5
    peak_ok = report.after.true_peak_dbfs <= (target_tp + 0.1)
    report.passed_qa = lufs_ok and peak_ok

    if not lufs_ok:
        report.warnings.append(
            f"⚠ Final LUFS {report.after.integrated_lufs:.1f} deviates from target {target_lufs:.1f}"
        )
    if not peak_ok:
        report.warnings.append(
            f"⚠ True peak {report.after.true_peak_dbfs:.1f} dBFS exceeds ceiling {target_tp:.1f}"
        )

    return report