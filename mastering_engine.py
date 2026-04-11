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

MasterCraft — Professional Mastering Engine v3  (Phase 1 Upgrade)
==================================================================
Phase 1 adds on top of the original 10-stage chain:

  NEW Stage 4.5 — Vocal humanization
      • Formant-aware de-essing (adaptive threshold from air_ratio)
      • Presence restoration (AI-tuned peak @ 3–5 kHz)
      • Breath/air shelf (12 kHz+)
      • Hollow-body anti-mud notch (320 Hz, fires if low_mid_ratio > 0.25)
      • Harshness trap (3.2 kHz surgical cut if upper_mid_ratio > 0.12)

  NEW Stage 6 — Transient shaper (between multiband comp and M/S)
      • Per-sample envelope follower (fast attack, slow release)
      • Attack boost for percussive / low-crest tracks
      • Sustain reduction for dense AI transient artifacts
      • Gain-neutral (RMS makeup after shaping)

  UPGRADED Stage 5 — Multiband compressor replaces single glue comp
      • 4 bands: Sub (<100 Hz), Low-mid (100-500 Hz), Mid (500-5k Hz), Air (5k+)
      • Crossovers: 4th-order Linkwitz-Riley (two 2nd-order Butterworth cascades)
      • Per-band threshold/ratio derived from analysis
      • Original glue comp kept as Stage 5.5 (lighter role)

  UPGRADED Stage 9 — Harmonic saturation (tape / tube / clip models)
      • Tape: even harmonics, asymmetric, warm — for acoustic/pop/R&B
      • Tube: odd harmonics (tanh), present — for rock/soul
      • Clip: hard waveshaper, edge — for EDM/hip-hop/electronic
      • Blend weights from acoustic_or_electronic field
      • RMS-neutral after blending

"""


from __future__ import annotations
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import librosa
import warnings
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import pedalboard as pb
import time
from scipy.signal import butter, sosfilt, sosfiltfilt

from audio_analyser import analyse_track, to_json, RichAudioAnalysis
from ai_advisor import get_mastering_recipe, MasteringRecipe, EQBand

warnings.filterwarnings("ignore")

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


@dataclass
class MasteringReport:
    input_path:       str = ""
    output_path:      str = ""
    platform:         str = "spotify"
    target_lufs:      float = -14.0
    target_true_peak: float = -1.0

    before: RichAudioAnalysis = field(default_factory=RichAudioAnalysis)
    after:  RichAudioAnalysis = field(default_factory=RichAudioAnalysis)

    recipe: Optional[MasteringRecipe] = None

    corrective_eq_applied:   list = field(default_factory=list)
    vocal_eq_applied:        list = field(default_factory=list)
    transient_attack_db:     float = 0.0
    transient_sustain_db:    float = 0.0
    mb_comp_gr_db:           dict  = field(default_factory=dict)
    glue_comp_gr_db:         float = 0.0
    ms_side_db:              float = 0.0
    tonal_eq_applied:        list  = field(default_factory=list)
    saturation_drive:        float = 0.0
    saturation_model:        str   = "tanh"
    limiter_gain_db:         float = 0.0

    warnings:   list = field(default_factory=list)
    passed_qa:  bool = False
    elapsed_s:  float = 0.0


def _build_eq_board(bands):
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
        else:
            plugins.append(pb.PeakFilter(cutoff_frequency_hz=hz, gain_db=db, q=q))
    return pb.Pedalboard(plugins) if plugins else None


# ═══ ORIGINAL STAGES (preserved) ═════════════════════════════════════════════

def _stage_dc_fix(audio, sr):
    sos = butter(2, 5, btype='high', fs=sr, output='sos')
    return sosfilt(sos, audio, axis=0).astype(np.float32)


def _stage_corrective_eq(audio, sr, recipe, report):
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


def _stage_glue_compression(audio, sr, recipe, report):
    meter = pyln.Meter(sr)
    lufs_before = float(meter.integrated_loudness(audio))
    comp = recipe.comp
    board = pb.Pedalboard([pb.Compressor(
        threshold_db=float(comp.threshold_db),
        ratio=float(np.clip(comp.ratio, 1.0, 4.0)),
        attack_ms=float(np.clip(comp.attack_ms, 1.0, 100.0)),
        release_ms=float(np.clip(comp.release_ms, 50.0, 1000.0)),
    )])
    result = board(audio.T.astype(np.float32), sr).T
    lufs_after = float(meter.integrated_loudness(result))
    gr_db = lufs_before - lufs_after
    report.glue_comp_gr_db = round(gr_db, 2)
    total_makeup = gr_db + float(comp.makeup_gain_db)
    if total_makeup > 0.1:
        result = (result * 10 ** (total_makeup / 20.0)).astype(np.float32)
    return result


def _stage_ms_processing(audio, sr, recipe, report):
    if audio.ndim < 2 or audio.shape[1] != 2:
        return audio
    L, R = audio[:, 0].copy(), audio[:, 1].copy()
    M = (L + R) * 0.5
    S = (L - R) * 0.5
    cutoff = float(np.clip(recipe.ms.low_mono_cutoff_hz, 60, 400))
    sos_hp = butter(4, cutoff, btype='high', fs=sr, output='sos')
    S_high = sosfiltfilt(sos_hp, S).astype(np.float32)
    side_boost = float(np.clip(recipe.ms.side_boost_db, -6.0, 6.0))
    S_final = S_high * 10 ** (side_boost / 20.0)
    report.ms_side_db = side_boost
    return np.stack([(M + S_final).astype(np.float32), (M - S_final).astype(np.float32)], axis=1)


def _stage_tonal_eq(audio, sr, recipe, report):
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


def _stage_normalize(audio, sr, target_lufs, target_tp, report):
    meter = pyln.Meter(sr)
    ceiling_linear = 10 ** (target_tp / 20.0)
    current_lufs = float(meter.integrated_loudness(audio))
    if not np.isfinite(current_lufs) or current_lufs < -70:
        current_lufs = -23.0
    gain_db = float(np.clip(target_lufs - current_lufs, -30.0, 30.0))
    audio = (audio * 10 ** (gain_db / 20.0)).astype(np.float32)
    current_peak = float(np.abs(audio).max())
    if current_peak > ceiling_linear:
        limiter = pb.Limiter(threshold_db=target_tp, release_ms=10)
        audio = limiter(audio.T, sr).T
        reduction_db = float(20 * np.log10(current_peak / (np.abs(audio).max() + 1e-10)))
    else:
        reduction_db = 0.0
    report.limiter_gain_db = round(gain_db - reduction_db, 2)
    return audio


def _post_chain_trim(audio, sr, target_lufs, target_tp, report):
    meter = pyln.Meter(sr)
    final_lufs = float(meter.integrated_loudness(audio))
    ceiling_linear = 10 ** (target_tp / 20.0)
    if np.isfinite(final_lufs) and final_lufs > -70:
        trim_db = target_lufs - final_lufs
        if abs(trim_db) > 0.3:
            candidate = (audio * 10 ** (trim_db / 20.0)).astype(np.float32)
            if float(np.abs(candidate).max()) <= ceiling_linear * 1.02:
                audio = candidate
                report.limiter_gain_db = round(report.limiter_gain_db + trim_db, 2)
            else:
                peak_after = float(np.abs(candidate).max())
                if peak_after > 1e-6:
                    new_thresh = target_tp - float(20 * np.log10(peak_after / ceiling_linear))
                    limiter = pb.Limiter(threshold_db=new_thresh, release_ms=10)
                    audio = limiter(candidate.T, sr).T
                    report.limiter_gain_db = round(report.limiter_gain_db + trim_db, 2)
    return np.clip(audio, -ceiling_linear, ceiling_linear).astype(np.float32)


# ═══ PHASE 1 — NEW / UPGRADED STAGES ════════════════════════════════════════

def _stage_vocal_humanization(audio, sr, analysis, recipe, report):
    """
    Targets AI vocal artifacts:
    1. Adaptive de-essing (scales with measured air_ratio)
    2. Presence restoration at 4 kHz (vocal sweet spot)
    3. Anti-mud notch at 320 Hz (fires when low_mid_ratio > 0.25)
    4. Harshness trap at 3.2 kHz (fires when upper_mid_ratio > 0.12)
    5. Air shelf at 12 kHz (always applied, amount scales with air_ratio)
    """
    vocal_presence = recipe.vocal_presence.lower()
    air_ratio    = float(analysis.air_ratio)
    low_mid_ratio = float(analysis.low_mid_ratio)
    upper_mid_ratio = float(analysis.upper_mid_ratio)

    applied = []

    # 1. De-essing
    if air_ratio > 0.08:
        strength = float(np.clip((air_ratio - 0.08) / 0.12, 0.0, 1.0))
        de_ess_db = -(1.5 + strength * 3.0)
        applied.append(EQBand(hz=8000.0, db=de_ess_db, q=2.0, type="peak",
                              reason=f"de-essing (air={air_ratio:.2f})"))

    # 2. Presence restoration (skip if vocals absent)
    if vocal_presence not in ("none", "unknown"):
        boost_map = {"low": 0.8, "medium": 1.5, "high": 2.0}
        boost = boost_map.get(vocal_presence, 1.2)
        applied.append(EQBand(hz=4000.0, db=boost, q=1.8, type="peak",
                              reason=f"vocal presence (+{boost:.1f}dB)"))

    # 3. Anti-mud notch
    if low_mid_ratio > 0.25:
        cut = float(np.clip(-(1.0 + (low_mid_ratio - 0.25) * 8.0), -4.5, -0.5))
        applied.append(EQBand(hz=320.0, db=cut, q=1.2, type="peak",
                              reason=f"mud reduction (low_mid={low_mid_ratio:.2f})"))

    # 4. Harshness trap
    if upper_mid_ratio > 0.12:
        cut = float(np.clip(-(1.0 + (upper_mid_ratio - 0.12) * 10.0), -4.0, -0.5))
        applied.append(EQBand(hz=3200.0, db=cut, q=2.5, type="peak",
                              reason=f"harshness trap (upper_mid={upper_mid_ratio:.2f})"))

    # 5. Air shelf (always)
    air_boost = float(np.clip(2.5 - air_ratio * 12.0, 0.5, 2.5))
    applied.append(EQBand(hz=12000.0, db=air_boost, q=0.7, type="highshelf",
                          reason=f"air shelf (+{air_boost:.1f}dB)"))

    if not applied:
        return audio

    board = _build_eq_board(applied)
    if not board:
        return audio

    result = board(audio.T.astype(np.float32), sr).T
    report.vocal_eq_applied = [
        {"hz": b.hz, "db": b.db, "q": b.q, "type": b.type, "reason": b.reason}
        for b in applied
    ]
    return result


def _stage_transient_shaper(audio, sr, analysis, recipe, report):
    """
    Envelope-follower transient shaper. Gain-neutral (RMS makeup).
    - Attack boost: adds punch to percussive / over-compressed tracks
    - Sustain reduction: tightens dense AI transient artifacts
    """
    perc = float(analysis.percussive_ratio)
    density = float(analysis.transient_density_hz)
    crest = float(analysis.crest_factor_db)

    # Attack gain: higher for percussive + low crest (already squashed)
    if perc > 0.5 and crest < 10:
        attack_db = float(np.clip(1.5 + (10 - crest) * 0.25, 0.5, 4.0))
    elif perc > 0.4:
        attack_db = 1.5
    else:
        attack_db = 0.5

    # Sustain gain: reduce for dense non-percussive (AI synth artifacts)
    if density > 4.0 and perc < 0.4:
        sustain_db = -1.5
    elif perc > 0.6:
        sustain_db = -1.0
    else:
        sustain_db = 0.0

    if abs(attack_db) < 0.2 and abs(sustain_db) < 0.2:
        return audio

    report.transient_attack_db  = round(attack_db, 2)
    report.transient_sustain_db = round(sustain_db, 2)

    mono = audio.mean(axis=1).astype(np.float64)
    n = len(mono)

    # Envelope follower: fast attack (1ms), slow release (100ms)
    a_att = np.exp(-1.0 / max(1, int(sr * 0.001)))
    a_rel = np.exp(-1.0 / max(1, int(sr * 0.100)))
    env = np.zeros(n)
    env[0] = abs(mono[0])
    for i in range(1, n):
        v = abs(mono[i])
        env[i] = v + (a_att if v > env[i-1] else a_rel) * (env[i-1] - v)
    env = np.maximum(env, 1e-10)

    # Normalised derivative → onset signal
    env_norm = env / (env.max() + 1e-10)
    deriv = np.diff(env_norm, prepend=env_norm[0])
    p95 = np.percentile(np.abs(deriv), 95) + 1e-10
    deriv = np.clip(deriv / p95, -1.0, 1.0)

    # Time-varying gain
    atk_lin = 10 ** (attack_db  / 20.0)
    sus_lin = 10 ** (sustain_db / 20.0)
    gain = np.ones(n)
    gain = np.where(deriv >  0.10, atk_lin, gain)
    gain = np.where(deriv < -0.05, sus_lin, gain)

    # Smooth (2ms) to prevent zipper noise
    w = max(1, int(sr * 0.002))
    gain = np.convolve(gain, np.ones(w)/w, mode='same')
    gain = np.clip(gain, 10**(-6/20), 10**(6/20))

    result = np.stack([
        (audio[:, 0] * gain).astype(np.float32),
        (audio[:, 1] * gain).astype(np.float32),
    ], axis=1)

    # Loudness-neutral makeup
    rms_in  = float(np.sqrt(np.mean(mono**2)) + 1e-10)
    rms_out = float(np.sqrt(np.mean(result.mean(axis=1)**2)) + 1e-10)
    result = (result * float(np.clip(rms_in / rms_out, 0.5, 2.0))).astype(np.float32)
    return result


def _stage_multiband_compression(audio, sr, analysis, recipe, report):
    """
    4-band Linkwitz-Riley compressor.
    Crossovers: 100 Hz | 500 Hz | 5000 Hz
    Bands:  Sub | Low-mid | Mid | Air
    Bands summed after per-band compression. Loudness-neutral makeup.
    """
    meter = pyln.Meter(sr)
    lufs_in = float(meter.integrated_loudness(audio))

    sub_r = float(analysis.sub_bass_ratio)
    air_r = float(analysis.air_ratio)
    bpm   = float(analysis.detected_bpm) or 120.0
    half_beat_ms = (60000.0 / max(60.0, bpm)) * 0.5

    def lr4_lp(fc, sig):
        sos = butter(2, fc, btype='low',  fs=sr, output='sos')
        return sosfiltfilt(sos, sosfiltfilt(sos, sig, axis=0), axis=0).astype(np.float32)

    def lr4_hp(fc, sig):
        sos = butter(2, fc, btype='high', fs=sr, output='sos')
        return sosfiltfilt(sos, sosfiltfilt(sos, sig, axis=0), axis=0).astype(np.float32)

    lo   = lr4_lp(500.0, audio)
    hi   = lr4_hp(500.0, audio)
    b_sub = lr4_lp(100.0,  lo)
    b_lm  = lr4_hp(100.0,  lo)
    b_mid = lr4_lp(5000.0, hi)
    b_air = lr4_hp(5000.0, hi)

    def compress(band, thr, ratio, atk, rel):
        board = pb.Pedalboard([pb.Compressor(
            threshold_db=thr,
            ratio=float(np.clip(ratio, 1.0, 6.0)),
            attack_ms=float(np.clip(atk, 0.5, 80.0)),
            release_ms=float(np.clip(rel, 30.0, 800.0)),
        )])
        return board(band.T.astype(np.float32), sr).T

    c_sub = compress(b_sub, -20.0 if sub_r > 0.30 else -24.0,
                     2.5 if sub_r > 0.30 else 1.8, 5.0, 80.0)
    c_lm  = compress(b_lm,  -22.0, 1.4, 12.0, half_beat_ms)
    c_mid = compress(b_mid, -24.0, 1.3, 20.0, min(half_beat_ms * 2, 600.0))
    c_air = compress(b_air, -18.0 if air_r > 0.12 else -30.0,
                     2.0 if air_r > 0.12 else 1.2, 3.0, 50.0)

    def gr(b, a):
        rb = float(np.sqrt(np.mean(b**2)) + 1e-10)
        ra = float(np.sqrt(np.mean(a**2)) + 1e-10)
        return round(20 * np.log10(ra / rb), 2)

    report.mb_comp_gr_db = {
        "sub": gr(b_sub, c_sub), "low_mid": gr(b_lm, c_lm),
        "mid": gr(b_mid, c_mid), "air":     gr(b_air, c_air),
    }

    result = (c_sub + c_lm + c_mid + c_air).astype(np.float32)

    # Makeup
    lufs_out = float(meter.integrated_loudness(result))
    if np.isfinite(lufs_in) and np.isfinite(lufs_out) and lufs_out < -5:
        makeup_db = float(np.clip(lufs_in - lufs_out, -6.0, 6.0))
        result = (result * 10 ** (makeup_db / 20.0)).astype(np.float32)

    return result


def _stage_saturation(audio, recipe, report):
    """
    3-model saturation blend: tape (even harmonics), tube (odd, tanh), clip (hard edge).
    Blend weights from acoustic_or_electronic. RMS-neutral output.
    """
    drive = float(np.clip(recipe.saturation_drive, 0.0, 1.0))
    if drive < 0.03:
        return audio

    mode = recipe.acoustic_or_electronic.lower()
    if mode == "acoustic":
        w_tape, w_tube, w_clip = 0.80, 0.20, 0.00
    elif mode == "hybrid":
        w_tape, w_tube, w_clip = 0.30, 0.50, 0.20
    else:
        w_tape, w_tube, w_clip = 0.20, 0.30, 0.50

    report.saturation_model = f"{mode}(tape={w_tape:.0%}/tube={w_tube:.0%}/clip={w_clip:.0%})"
    report.saturation_drive = drive

    x = audio.astype(np.float64)

    def tape(x):
        d = 1.0 + drive * 3.0
        xi = x * d
        y_n = xi - (xi**3) / 3.0
        y_p = xi * 1.03 - (xi**3) / 3.0
        y = np.where(xi >= 0, y_p, y_n)
        pk = np.abs(y).max() + 1e-10
        return y / pk * np.abs(xi).max()

    def tube(x):
        d = 1.0 + drive * 5.0
        return np.tanh(d * x) / (np.tanh(d) + 1e-10)

    def clip(x):
        thr = 1.0 - drive * 0.5
        knee = 0.05
        abs_x, sgn = np.abs(x), np.sign(x)
        y = x.copy()
        in_k = (abs_x > thr) & (abs_x < thr + knee)
        over  = abs_x >= thr + knee
        y[in_k] = sgn[in_k] * (thr + knee * np.tanh((abs_x[in_k] - thr) / (knee + 1e-10)))
        y[over]  = sgn[over] * (thr + knee)
        return y

    result = np.zeros_like(x)
    if w_tape > 0.01: result += w_tape * tape(x)
    if w_tube > 0.01: result += w_tube * tube(x)
    if w_clip > 0.01: result += w_clip * clip(x)

    # RMS-neutral
    rms_in  = float(np.sqrt(np.mean(x**2))      + 1e-10)
    rms_out = float(np.sqrt(np.mean(result**2)) + 1e-10)
    return (result * (rms_in / rms_out)).astype(np.float32)


# ═══ MAIN ════════════════════════════════════════════════════════════════════

def master_track(
    input_path, output_path, platform="spotify",
    api_key=None, custom_lufs=None, custom_true_peak=None,
    force_fallback=False,
):
    """
    Full AI-driven mastering chain — Phase 1 (13 stages).

      1.  Load + stereo + resample to 44100
      2.  Rich analysis
      3.  AI advisor → recipe
      3.5 DC fix
      3.7 Noise gate (conditional)
      4.  Corrective EQ
      4.5 Vocal humanization  ← NEW
      5.  Multiband compression (4-band LR4)  ← UPGRADED
      5.5 Glue compression (lighter, BPM-locked)
      6.  Transient shaper  ← NEW
      7.  M/S processing
      8.  Tonal EQ
      9.  Harmonic saturation (tape/tube/clip)  ← UPGRADED
      10. Loudness normalisation + true-peak limiter
      11. Post-chain verification trim
      12. Write 24-bit WAV
    """
    t0 = time.time()
    input_path  = Path(input_path)
    output_path = Path(output_path)

    if platform in PLATFORM_TARGETS:
        target_lufs = custom_lufs  or PLATFORM_TARGETS[platform]["lufs"]
        target_tp   = custom_true_peak or PLATFORM_TARGETS[platform]["true_peak"]
    elif custom_lufs and custom_true_peak:
        target_lufs, target_tp = custom_lufs, custom_true_peak
    else:
        raise ValueError(f"Unknown platform '{platform}'. Options: {list(PLATFORM_TARGETS)}")

    report = MasteringReport(
        input_path=str(input_path), output_path=str(output_path),
        platform=platform, target_lufs=target_lufs, target_true_peak=target_tp,
    )

    # Load
    audio, sr = sf.read(str(input_path), dtype="float32", always_2d=True)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    elif audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if sr != 44100:
        L = librosa.resample(audio[:, 0], orig_sr=sr, target_sr=44100)
        R = librosa.resample(audio[:, 1], orig_sr=sr, target_sr=44100)
        audio = np.stack([L, R], axis=1).astype(np.float32)
        sr = 44100

    print("  [1/12] Analysing track...", flush=True)
    analysis = analyse_track(audio, sr, str(input_path))
    report.before = analysis
    analysis_json = to_json(analysis)

    if analysis.stereo_correlation < 0.3:
        report.warnings.append("⚠ Severe phase issues — check mono compatibility")
    if analysis.dynamic_range_db < 5:
        report.warnings.append("⚠ Very low dynamic range — may already be over-compressed")
    if analysis.noise_floor_db > -50:
        report.warnings.append(f"⚠ High noise floor at {analysis.noise_floor_db:.1f} dBFS")
    if analysis.low_end_mono < 0.7:
        report.warnings.append("⚠ Bass is not mono — may cause phase issues on mono playback")

    print("  [2/12] Consulting AI advisor...", flush=True)
    if force_fallback:
        from ai_advisor import _rule_based_fallback
        recipe = _rule_based_fallback(analysis_json, platform)
    else:
        recipe = get_mastering_recipe(analysis_json, platform=platform, api_key=api_key)
    report.recipe = recipe

    target_lufs = float(np.clip(recipe.target_lufs, -20.0, -6.0))
    target_tp   = float(np.clip(recipe.true_peak_dbfs, -3.0, -0.1))
    report.target_lufs      = target_lufs
    report.target_true_peak = target_tp

    print("  [3/12] DC offset fix...", flush=True)
    audio = _stage_dc_fix(audio, sr)

    if analysis.noise_floor_db > -50:
        print("  [3.5/12] Noise gate...", flush=True)
        gate = pb.NoiseGate(threshold_db=analysis.noise_floor_db + 6.0, attack_ms=5, release_ms=100)
        audio = gate(audio.T, sr).T
        report.warnings.append(f"⚠ Applied noise gate ({analysis.noise_floor_db+6:.1f} dBFS)")

    print(f"  [4/12] Corrective EQ ({len(recipe.corrective_eq)} bands)...", flush=True)
    audio = _stage_corrective_eq(audio, sr, recipe, report)

    print(f"  [4.5/12] Vocal humanization (presence={recipe.vocal_presence})...", flush=True)
    audio = _stage_vocal_humanization(audio, sr, analysis, recipe, report)

    print("  [5/12] Multiband compression (4-band LR4)...", flush=True)
    audio = _stage_multiband_compression(audio, sr, analysis, recipe, report)

    print(f"  [5.5/12] Glue compression ({recipe.comp.ratio:.1f}:1)...", flush=True)
    audio = _stage_glue_compression(audio, sr, recipe, report)

    print("  [6/12] Transient shaper...", flush=True)
    audio = _stage_transient_shaper(audio, sr, analysis, recipe, report)

    print(f"  [7/12] M/S processing (side {recipe.ms.side_boost_db:+.1f}dB)...", flush=True)
    audio = _stage_ms_processing(audio, sr, recipe, report)

    print(f"  [8/12] Tonal EQ ({len(recipe.tonal_eq)} bands)...", flush=True)
    audio = _stage_tonal_eq(audio, sr, recipe, report)

    print(f"  [9/12] Harmonic saturation ({recipe.acoustic_or_electronic})...", flush=True)
    audio = _stage_saturation(audio, recipe, report)

    print(f"  [10/12] Normalising to {target_lufs:.0f} LUFS / {target_tp:.1f} dBTP...", flush=True)
    audio = _stage_normalize(audio, sr, target_lufs, target_tp, report)

    print("  [11/12] Verification trim...", flush=True)
    audio = _post_chain_trim(audio, sr, target_lufs, target_tp, report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr, subtype="PCM_24")

    report.after = analyse_track(audio, sr, str(output_path))
    report.elapsed_s = round(time.time() - t0, 1)

    lufs_ok = abs(report.after.integrated_lufs - target_lufs) < 1.5
    peak_ok = report.after.true_peak_dbfs <= (target_tp + 0.1)
    report.passed_qa = lufs_ok and peak_ok

    if not lufs_ok:
        report.warnings.append(
            f"⚠ Final LUFS {report.after.integrated_lufs:.1f} deviates from target {target_lufs:.1f}")
    if not peak_ok:
        report.warnings.append(
            f"⚠ True peak {report.after.true_peak_dbfs:.1f} dBFS exceeds ceiling {target_tp:.1f}")

    print(f"  [12/12] Done — {report.elapsed_s:.1f}s  QA: {'PASS' if report.passed_qa else 'FAIL'}",
          flush=True)
    return report
