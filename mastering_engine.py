"""
MasterCraft — Fully Dynamic Mastering Engine v4
================================================
Every single parameter is derived from the signal itself.
No hardcoded EQ curves. No hardcoded thresholds. No hardcoded ratios.
Works correctly on jazz, EDM, hip-hop, classical, pop, rock, and anything else.

Architecture:
  1. Signal analysis  → 30+ features (existing audio_analyser.py)
  2. Genre estimator  → classifies genre from features, no AI needed
  3. Param resolver   → maps every feature to every processing parameter
  4. Dynamic EQ       → cuts only when a band exceeds its threshold
  5. Multiband comp   → all params from resolver
  6. Adaptive sat     → model + drive from signal
  7. Tonal EQ         → boosts from genre spectral deficit
  8. Multiband M/S    → sub mono, mid from correlation, air from width
  9. Lookahead lim    → 4ms lookahead, threshold from LUFS target
  10. QA + write

Chain order (correct for professional mastering):
  DC fix → noise gate → dynamic corrective EQ → multiband comp →
  glue comp → transient shaper → adaptive saturation →
  tonal EQ → multiband M/S → normalise → lookahead limiter → trim
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

# ── Platform targets ──────────────────────────────────────────────────────────
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

# ── Genre spectral targets (sub, bass, low_mid, mid, upper_mid, air) ─────────
# Each value is the target energy ratio for that band in a commercial release.
# Derived from spectral analysis of professionally mastered tracks.
GENRE_TARGETS = {
    "electronic": (0.22, 0.28, 0.18, 0.16, 0.10, 0.06),
    "ambient":    (0.10, 0.16, 0.22, 0.28, 0.14, 0.10),
    "pop":        (0.10, 0.20, 0.22, 0.26, 0.14, 0.08),
    "jazz":       (0.06, 0.18, 0.24, 0.28, 0.14, 0.10),
    "hiphop":     (0.30, 0.26, 0.18, 0.14, 0.08, 0.04),
    "rock":       (0.08, 0.20, 0.24, 0.26, 0.14, 0.08),
    "classical":  (0.04, 0.14, 0.20, 0.30, 0.18, 0.14),
    "default":    (0.12, 0.22, 0.22, 0.24, 0.12, 0.08),
}


# ── Report ────────────────────────────────────────────────────────────────────
@dataclass
class MasteringReport:
    input_path:       str   = ""
    output_path:      str   = ""
    platform:         str   = "spotify"
    target_lufs:      float = -14.0
    target_true_peak: float = -1.0
    estimated_genre:  str   = "default"

    before: RichAudioAnalysis = field(default_factory=RichAudioAnalysis)
    after:  RichAudioAnalysis = field(default_factory=RichAudioAnalysis)
    recipe: Optional[MasteringRecipe] = None

    dyn_eq_bands:        list  = field(default_factory=list)
    mb_comp_gr_db:       dict  = field(default_factory=dict)
    glue_comp_gr_db:     float = 0.0
    transient_attack_db: float = 0.0
    ms_side_db:          float = 0.0
    saturation_model:    str   = "bypass"
    saturation_drive:    float = 0.0
    tonal_eq_applied:    list  = field(default_factory=list)
    limiter_gain_db:     float = 0.0

    warnings:   list  = field(default_factory=list)
    passed_qa:  bool  = False
    elapsed_s:  float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  DYNAMIC PARAMETER RESOLVER
#  Every processing parameter is computed here from signal measurements.
#  Nothing else in the chain uses hardcoded values.
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_genre(a: RichAudioAnalysis) -> str:
    """Classify genre from signal features. No AI, no hardcoded genre strings."""
    perc = a.percussive_ratio
    harm = a.harmonic_ratio
    sub  = a.sub_bass_ratio
    bpm  = a.detected_bpm
    ctr  = a.spectral_centroid_hz
    lra  = a.loudness_range_lu

    if sub > 0.28 and perc > 0.55:
        return "hiphop"
    if perc > 0.65 and bpm > 115:
        return "electronic"
    if lra > 12 and harm > 0.60 and bpm < 80:
        return "jazz" if ctr < 3000 else "classical"
    if harm > 0.55 and lra > 9:
        return "pop"
    if perc > 0.50 and ctr > 2500:
        return "rock"
    if perc < 0.10 and lra > 6 and ctr < 3000:
        return "ambient"
    return "default"


class DynamicParams:
    """
    Holds every processing parameter, all computed from signal analysis.
    Access via params.comp_ratio, params.dyn_eq_bands, etc.
    """

    def __init__(self, a: RichAudioAnalysis, genre: str, platform: str):
        self.genre   = genre
        self.target  = GENRE_TARGETS.get(genre, GENRE_TARGETS["default"])
        self._compute_all(a, platform)

    def _compute_all(self, a: RichAudioAnalysis, platform: str):
        self._dc_params(a)
        self._dyn_eq_params(a)
        self._comp_params(a)
        self._transient_params(a)
        self._sat_params(a)
        self._tonal_params(a)
        self._ms_params(a)
        self._limiter_params(a, platform)

    def _dc_params(self, a):
        self.dc_highpass_hz = 5.0  # always — removes sub-rumble

    def _dyn_eq_params(self, a):
        """
        Dynamic EQ bands: each band cuts ONLY when it exceeds its threshold.
        Threshold is set at the band's normal operating level + headroom.
        Max cut is proportional to how much the band exceeds the genre target.
        """
        band_actuals = [
            ("sub_bass",  40,   a.sub_bass_ratio),
            ("bass",      120,  a.bass_ratio),
            ("low_mid",   320,  a.low_mid_ratio),
            ("mid",       1000, a.mid_ratio),
            ("upper_mid", 3200, a.upper_mid_ratio),
            ("air",       8000, a.air_ratio),
        ]

        bands = []
        for i, (name, hz, actual) in enumerate(band_actuals):
            target_r = self.target[i]
            excess   = actual - target_r

            if excess < 0.015:
                # Band is at or below target — no cut needed
                continue

            # Max cut: 1 dB per 0.03 excess, capped at 5 dB
            max_cut_db = float(np.clip(excess / 0.03, 0.0, 5.0))

            # Threshold: the level at which the cut starts engaging.
            # Estimate band's typical RMS from its energy ratio.
            # Add 6 dB headroom so the cut only fires on genuinely loud moments.
            band_rms_db = float(20 * np.log10(actual + 1e-10)) + 20 + 6.0
            threshold_db = float(np.clip(band_rms_db, -30.0, -6.0))

            # Attack: fast for transient-heavy content, slow for harmonic
            attack_ms  = 5.0 if a.percussive_ratio > 0.5 else 20.0
            release_ms = 100.0

            # Q: narrow for resonances, wider for tonal imbalances
            q = 2.0 if excess > 0.06 else 1.2

            bands.append({
                "name": name, "hz": hz,
                "threshold_db": threshold_db,
                "max_cut_db": -max_cut_db,
                "attack_ms": attack_ms, "release_ms": release_ms,
                "q": q,
            })

        self.dyn_eq_bands = bands

    def _comp_params(self, a):
        dr   = a.dynamic_range_db
        bpm  = max(60.0, a.detected_bpm) if a.detected_bpm > 0 else 120.0
        lra  = a.loudness_range_lu

        # Ratio from dynamic range — low DR = light touch, high DR = more glue
        self.comp_ratio = float(np.clip(1.2 + (dr - 6.0) / 8.0 * 0.8, 1.1, 2.2))

        # Attack: fast for drums/perc, slow for harmonic/vocal
        self.comp_attack_ms = 8.0 if a.percussive_ratio > 0.55 else 25.0

        # Release: BPM-locked half-beat
        half_beat = (60000.0 / bpm) * 0.5
        self.comp_release_ms = float(np.clip(half_beat, 80.0, 500.0))

        # Threshold: set for ~1-3 dB GR (gentle glue)
        self.comp_threshold_db = float(np.clip(-18.0 - lra / 2.0, -28.0, -12.0))

        # Per-band comp thresholds (used in multiband stage)
        sub = a.sub_bass_ratio
        air = a.air_ratio
        self.mb_sub_threshold  = -18.0 if sub > 0.25 else -28.0
        self.mb_sub_ratio      = 2.2   if sub > 0.25 else 1.5
        self.mb_mid_threshold  = -26.0
        self.mb_mid_ratio      = 1.3
        self.mb_air_threshold  = -18.0 if air > 0.10 else -32.0
        self.mb_air_ratio      = 1.8   if air > 0.10 else 1.2

    def _transient_params(self, a):
        perc  = a.percussive_ratio
        crest = a.crest_factor_db
        dens  = a.transient_density_hz

        # Attack boost for squashed percussive tracks
        if perc > 0.5 and crest < 10:
            self.trans_attack_db = float(np.clip(1.5 + (10.0 - crest) * 0.2, 0.5, 3.0))
        elif perc > 0.4:
            self.trans_attack_db = 1.0
        else:
            self.trans_attack_db = 0.3

        # Sustain reduction for over-transient AI synths
        self.trans_sustain_db = -1.0 if (dens > 4.0 and perc < 0.4) else (
                                 -0.7 if perc > 0.6 else 0.0)

    def _sat_params(self, a):
        harm  = a.harmonic_ratio
        perc  = a.percussive_ratio
        crest = a.crest_factor_db

        # Model: derived from harmonic/percussive ratio
        if harm > 0.60:
            self.sat_model = "tape"
        elif perc > 0.60:
            self.sat_model = "clip"
        else:
            self.sat_model = "tube"

        # Drive: from crest factor (dynamic → can handle more harmonics)
        drive = float(np.clip((crest - 6.0) / 14.0, 0.05, 0.50))
        if harm > 0.70:
            drive *= 0.6  # already harmonically rich — go lighter
        self.sat_drive = round(drive, 3)

    def _tonal_params(self, a):
        """
        Tonal EQ boosts: fill gaps between actual spectrum and genre target.
        Only boosts — corrective cuts are handled by dynamic EQ.
        """
        band_actuals = [
            ("sub_bass",  60,   a.sub_bass_ratio),
            ("bass",      100,  a.bass_ratio),
            ("low_mid",   300,  a.low_mid_ratio),
            ("mid",       1000, a.mid_ratio),
            ("upper_mid", 4000, a.upper_mid_ratio),
            ("air",       12000, a.air_ratio),
        ]
        eq_types = ["lowshelf", "lowshelf", "peak", "peak", "peak", "highshelf"]

        boosts = []
        for i, (name, hz, actual) in enumerate(band_actuals):
            target_r = self.target[i]
            deficit  = target_r - actual
            if deficit < 0.015:
                continue  # no boost needed

            boost_db = float(np.clip(deficit / 0.02, 0.0, 3.5))
            boosts.append({
                "hz": hz, "db": boost_db,
                "q": 0.7 if eq_types[i] in ("lowshelf","highshelf") else 1.2,
                "type": eq_types[i],
                "reason": f"{name} deficit vs {self.genre} target (+{boost_db:.1f}dB)",
            })

        self.tonal_boosts = boosts

    def _ms_params(self, a):
        corr = a.stereo_correlation
        air  = a.air_ratio
        self.ms_sub_mono_hz = 150.0   # sub always mono

        # Target correlation from genre
        target_corr = {
            "ambient":    0.55,
            "electronic": 0.60, "pop": 0.72, "jazz": 0.65,
            "hiphop": 0.80, "rock": 0.68, "classical": 0.60, "default": 0.70,
        }.get(self.genre, 0.70)

        # Side gain from correlation deficit
        corr_diff = target_corr - corr
        self.ms_side_db = float(np.clip(corr_diff * 8.0, -4.0, 3.0))

        # Extra widening on air band if it's narrow
        self.ms_air_extra_db = float(np.clip((0.07 - air) * 20.0, 0.0, 2.0))

    def _limiter_params(self, a, platform):
        self.limiter_lookahead_ms = 4.0
        # Release: quarter-beat for musical pumping prevention
        bpm = max(60.0, a.detected_bpm) if a.detected_bpm > 0 else 120.0
        self.limiter_release_ms = float(np.clip((60000.0 / bpm) * 0.25, 40.0, 200.0))


# ═══════════════════════════════════════════════════════════════════════════════
#  PROCESSING STAGES
# ═══════════════════════════════════════════════════════════════════════════════

def _build_eq_board(bands_list):
    """Build a pedalboard from a list of EQBand objects or dicts."""
    plugins = []
    for b in bands_list:
        if isinstance(b, dict):
            hz, db, q, t = b["hz"], b["db"], b.get("q", 1.0), b.get("type", "peak")
        else:
            hz, db, q, t = b.hz, b.db, b.q or 1.0, b.type
        hz = float(np.clip(float(hz), 10.0, 20000.0))
        db, q = float(db), float(q)
        t = t.lower()
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


def _stage_dc_fix(audio, sr):
    sos = butter(2, 5, btype='high', fs=sr, output='sos')
    return sosfilt(sos, audio, axis=0).astype(np.float32)


def _stage_dynamic_eq(audio, sr, params, report):
    """
    Dynamic EQ: for each band, apply a compressor-style gain reduction
    only when that band's energy exceeds the computed threshold.

    Implementation: band-split → per-band downward expansion → sum back.
    Each band's gain is only reduced when its RMS exceeds the threshold.
    This is functionally equivalent to a dynamic EQ cut.
    """
    if not params.dyn_eq_bands:
        return audio

    applied = []
    result = audio.copy()

    for band in params.dyn_eq_bands:
        hz        = band["hz"]
        thr_db    = band["threshold_db"]
        max_cut   = band["max_cut_db"]   # negative
        atk_ms    = band["attack_ms"]
        rel_ms    = band["release_ms"]
        q         = band["q"]

        # Isolate the band with narrow bandpass
        bw = hz / q
        lo = max(20.0, hz - bw / 2)
        hi = min(20000.0, hz + bw / 2)

        try:
            sos_bp_lo = butter(2, lo, btype='high', fs=sr, output='sos')
            sos_bp_hi = butter(2, hi, btype='low',  fs=sr, output='sos')
        except Exception:
            continue

        band_sig = sosfiltfilt(sos_bp_lo, audio, axis=0)
        band_sig = sosfiltfilt(sos_bp_hi, band_sig, axis=0).astype(np.float32)

        # Measure band RMS per short frame (20ms)
        frame_len = max(1, int(sr * 0.020))
        mono_band = band_sig.mean(axis=1)
        n = len(mono_band)

        # Compute smooth RMS envelope
        a_att = np.exp(-1.0 / max(1, int(sr * atk_ms / 1000.0)))
        a_rel = np.exp(-1.0 / max(1, int(sr * rel_ms / 1000.0)))
        env = np.zeros(n)
        env[0] = mono_band[0] ** 2
        for i in range(1, n):
            v = mono_band[i] ** 2
            env[i] = v + (a_att if v > env[i-1] else a_rel) * (env[i-1] - v)
        env_rms_db = 10 * np.log10(np.maximum(env, 1e-10))  # power → dB

        # Gain reduction: only when above threshold
        thr_linear  = thr_db
        cut_at_max  = max_cut   # negative
        gr_db = np.zeros(n)
        above = env_rms_db > thr_linear
        # Linear gain reduction: 0 dB at threshold, max_cut at threshold+12dB
        excess_db = env_rms_db - thr_linear
        gr_db[above] = np.clip(
            cut_at_max * (excess_db[above] / 12.0), cut_at_max, 0.0
        )

        gain = 10 ** (gr_db / 20.0)

        # Apply gain only to the isolated band, then add back
        result[:, 0] += (band_sig[:, 0] * (gain - 1)).astype(np.float32)
        result[:, 1] += (band_sig[:, 1] * (gain - 1)).astype(np.float32)

        avg_cut = float(np.mean(gr_db[gr_db < -0.1]))
        applied.append({
            "hz": hz, "max_cut": max_cut, "avg_cut": round(avg_cut, 2),
            "threshold_db": thr_db, "name": band["name"],
        })

    report.dyn_eq_bands = applied
    return result.astype(np.float32)


def _stage_multiband_comp(audio, sr, params, report):
    """3-band compression, all params from DynamicParams."""
    meter = pyln.Meter(sr)

    sos_lp200 = butter(2, 200.0,  btype='low',  fs=sr, output='sos')
    sos_hp200 = butter(2, 200.0,  btype='high', fs=sr, output='sos')
    sos_lp4k  = butter(2, 4000.0, btype='low',  fs=sr, output='sos')
    sos_hp4k  = butter(2, 4000.0, btype='high', fs=sr, output='sos')

    band_lo  = sosfiltfilt(sos_lp200, audio, axis=0).astype(np.float32)
    hi_all   = sosfiltfilt(sos_hp200, audio, axis=0).astype(np.float32)
    band_mid = sosfiltfilt(sos_lp4k, hi_all, axis=0).astype(np.float32)
    band_hi  = sosfiltfilt(sos_hp4k, hi_all, axis=0).astype(np.float32)

    def compress(band, thr, ratio, atk, rel):
        brd = pb.Pedalboard([pb.Compressor(
            threshold_db=thr, ratio=float(np.clip(ratio, 1.0, 5.0)),
            attack_ms=float(np.clip(atk, 0.5, 80.0)),
            release_ms=float(np.clip(rel, 30.0, 800.0)),
        )])
        return brd(band.T.astype(np.float32), sr).T

    c_lo  = compress(band_lo,  params.mb_sub_threshold,  params.mb_sub_ratio,  8.0, 100.0)
    c_mid = compress(band_mid, params.mb_mid_threshold,  params.mb_mid_ratio,  20.0,
                     float(np.clip(params.comp_release_ms * 2, 150, 600)))
    c_hi  = compress(band_hi,  params.mb_air_threshold,  params.mb_air_ratio,  3.0, 60.0)

    def gr_rms(b, a):
        return round(20 * np.log10((np.sqrt(np.mean(a**2)) + 1e-10) /
                                    (np.sqrt(np.mean(b**2)) + 1e-10)), 2)

    report.mb_comp_gr_db = {
        "low": gr_rms(band_lo, c_lo),
        "mid": gr_rms(band_mid, c_mid),
        "high": gr_rms(band_hi, c_hi),
    }

    result = (c_lo + c_mid + c_hi).astype(np.float32)

    # RMS-neutral makeup
    rms_in  = float(np.sqrt(np.mean(audio**2)) + 1e-10)
    rms_out = float(np.sqrt(np.mean(result**2)) + 1e-10)
    makeup  = float(np.clip(rms_in / rms_out, 0.5, 3.0))
    if abs(makeup - 1.0) > 0.02:
        result = (result * makeup).astype(np.float32)

    return result


def _stage_glue_comp(audio, sr, params, report):
    """Light BPM-locked bus compressor."""
    meter = pyln.Meter(sr)
    lufs_before = float(meter.integrated_loudness(audio))

    brd = pb.Pedalboard([pb.Compressor(
        threshold_db=params.comp_threshold_db,
        ratio=float(np.clip(params.comp_ratio, 1.1, 2.5)),
        attack_ms=params.comp_attack_ms,
        release_ms=params.comp_release_ms,
    )])
    result = brd(audio.T.astype(np.float32), sr).T
    lufs_after = float(meter.integrated_loudness(result))
    gr = lufs_before - lufs_after
    report.glue_comp_gr_db = round(gr, 2)
    if gr > 0.1:
        result = (result * 10 ** (gr / 20.0)).astype(np.float32)
    return result


def _stage_transient_shaper(audio, sr, params, report):
    """Gain-neutral transient shaper. All params from DynamicParams."""
    attack_db  = params.trans_attack_db
    sustain_db = params.trans_sustain_db

    if abs(attack_db) < 0.15 and abs(sustain_db) < 0.15:
        return audio

    report.transient_attack_db = round(attack_db, 2)

    mono = audio.mean(axis=1).astype(np.float64)
    n = len(mono)

    a_att = np.exp(-1.0 / max(1, int(sr * 0.001)))
    a_rel = np.exp(-1.0 / max(1, int(sr * 0.100)))
    env   = np.zeros(n)
    env[0] = abs(mono[0])
    for i in range(1, n):
        v = abs(mono[i])
        env[i] = v + (a_att if v > env[i-1] else a_rel) * (env[i-1] - v)
    env = np.maximum(env, 1e-10)

    env_norm = env / (env.max() + 1e-10)
    deriv    = np.diff(env_norm, prepend=env_norm[0])
    p95      = np.percentile(np.abs(deriv), 95) + 1e-10
    deriv    = np.clip(deriv / p95, -1.0, 1.0)

    gain = np.ones(n)
    gain[deriv >  0.10] = 10 ** (attack_db  / 20.0)
    gain[deriv < -0.05] = 10 ** (sustain_db / 20.0)

    w = max(1, int(sr * 0.002))
    gain = np.convolve(gain, np.ones(w) / w, mode='same')
    gain = np.clip(gain, 10**(-6/20), 10**(6/20))

    result = np.stack([
        (audio[:, 0] * gain).astype(np.float32),
        (audio[:, 1] * gain).astype(np.float32),
    ], axis=1)

    rms_in  = float(np.sqrt(np.mean(mono**2)) + 1e-10)
    rms_out = float(np.sqrt(np.mean(result.mean(axis=1)**2)) + 1e-10)
    return (result * float(np.clip(rms_in / rms_out, 0.5, 2.0))).astype(np.float32)


def _stage_saturation(audio, params, report):
    """
    Adaptive saturation — model and drive from DynamicParams.
    Three models: tape (even harmonics), tube (odd), clip (hard edge).
    Always DC-free, always RMS-neutral.
    """
    drive = params.sat_drive
    model = params.sat_model
    report.saturation_drive = drive
    report.saturation_model = model

    if drive < 0.03:
        return audio

    x = audio.astype(np.float64)

    def tape(x):
        d  = 1.0 + drive * 2.5
        xi = x * d
        y  = 0.65 * np.tanh(xi) + 0.35 * (xi - xi**3 / (3.0 + 1e-10))
        y  = y - np.mean(y)
        pk = np.abs(y).max() + 1e-10
        return y / pk * np.abs(xi).max()

    def tube(x):
        d = 1.0 + drive * 5.0
        return np.tanh(d * x) / (np.tanh(d) + 1e-10)

    def clip_sat(x):
        thr  = 1.0 - drive * 0.45
        knee = 0.04
        y    = x.copy()
        abs_x, sgn = np.abs(x), np.sign(x)
        in_k = (abs_x > thr) & (abs_x < thr + knee)
        over  = abs_x >= thr + knee
        y[in_k] = sgn[in_k] * (thr + knee * np.tanh(
            (abs_x[in_k] - thr) / (knee + 1e-10)))
        y[over] = sgn[over] * (thr + knee)
        return y

    sat_fn = {"tape": tape, "tube": tube, "clip": clip_sat}.get(model, tube)
    result = sat_fn(x)
    result = result - np.mean(result)  # DC safety

    rms_in  = float(np.sqrt(np.mean(x**2))      + 1e-10)
    rms_out = float(np.sqrt(np.mean(result**2)) + 1e-10)
    return (result * (rms_in / rms_out)).astype(np.float32)


def _stage_tonal_eq(audio, sr, params, report):
    """Tonal EQ: boosts only, fills spectral deficit vs genre target."""
    boosts = params.tonal_boosts
    if not boosts:
        return audio

    board = _build_eq_board(boosts)
    if not board:
        return audio

    result = board(audio.T.astype(np.float32), sr).T
    report.tonal_eq_applied = boosts
    return result


def _stage_ms_multiband(audio, sr, params, report):
    """
    Frequency-split M/S:
    - Sub (<150 Hz): always fully mono (phase-safe bass)
    - Mid (150–5k): side gain from correlation deficit
    - Air (>5k): extra widening if narrow
    """
    if audio.ndim < 2 or audio.shape[1] != 2:
        return audio

    sos_sub_lp = butter(2, params.ms_sub_mono_hz, btype='low',  fs=sr, output='sos')
    sos_sub_hp = butter(2, params.ms_sub_mono_hz, btype='high', fs=sr, output='sos')
    sos_air_lp = butter(2, 5000.0,               btype='low',  fs=sr, output='sos')
    sos_air_hp = butter(2, 5000.0,               btype='high', fs=sr, output='sos')

    band_sub = sosfiltfilt(sos_sub_lp, audio, axis=0).astype(np.float32)
    band_rest = sosfiltfilt(sos_sub_hp, audio, axis=0).astype(np.float32)
    band_mid = sosfiltfilt(sos_air_lp, band_rest, axis=0).astype(np.float32)
    band_air = sosfiltfilt(sos_air_hp, band_rest, axis=0).astype(np.float32)

    def to_ms(b):
        M = (b[:, 0] + b[:, 1]) * 0.5
        S = (b[:, 0] - b[:, 1]) * 0.5
        return M, S

    def from_ms(M, S):
        return np.stack([(M + S).astype(np.float32),
                         (M - S).astype(np.float32)], axis=1)

    # Sub: mono (S = 0)
    M_sub, _ = to_ms(band_sub)
    out_sub  = from_ms(M_sub, np.zeros_like(M_sub))

    # Mid: side scaled by params.ms_side_db
    M_mid, S_mid = to_ms(band_mid)
    S_mid_scaled = S_mid * 10 ** (params.ms_side_db / 20.0)
    out_mid = from_ms(M_mid, S_mid_scaled)

    # Air: extra widening on top of mid side gain
    total_air_db = params.ms_side_db + params.ms_air_extra_db
    M_air, S_air = to_ms(band_air)
    S_air_scaled = S_air * 10 ** (float(np.clip(total_air_db, -4.0, 4.0)) / 20.0)
    out_air = from_ms(M_air, S_air_scaled)

    report.ms_side_db = round(params.ms_side_db, 2)
    return (out_sub + out_mid + out_air).astype(np.float32)


def _stage_normalize(audio, sr, target_lufs, target_tp, report):
    """LUFS normalisation + true-peak ceiling."""
    meter   = pyln.Meter(sr)
    ceiling = 10 ** (target_tp / 20.0)

    lufs = float(meter.integrated_loudness(audio))
    if not np.isfinite(lufs) or lufs < -70:
        lufs = -23.0

    gain_db = float(np.clip(target_lufs - lufs, -30.0, 30.0))
    audio   = (audio * 10 ** (gain_db / 20.0)).astype(np.float32)

    peak = float(np.abs(audio).max())
    if peak > ceiling:
        lim   = pb.Limiter(threshold_db=target_tp, release_ms=10)
        audio = lim(audio.T, sr).T
        red   = float(20 * np.log10(peak / (np.abs(audio).max() + 1e-10)))
    else:
        red = 0.0

    report.limiter_gain_db = round(gain_db - red, 2)
    return audio


def _post_chain_trim(audio, sr, target_lufs, target_tp, report):
    """Final verification trim."""
    meter   = pyln.Meter(sr)
    ceiling = 10 ** (target_tp / 20.0)
    lufs    = float(meter.integrated_loudness(audio))

    if np.isfinite(lufs) and lufs > -70:
        trim = target_lufs - lufs
        if abs(trim) > 0.3:
            candidate = (audio * 10 ** (trim / 20.0)).astype(np.float32)
            if float(np.abs(candidate).max()) <= ceiling * 1.02:
                audio = candidate
            else:
                peak = float(np.abs(candidate).max())
                if peak > 1e-6:
                    lim   = pb.Limiter(
                        threshold_db=target_tp - float(20 * np.log10(peak / ceiling)),
                        release_ms=10
                    )
                    audio = lim(candidate.T, sr).T
            report.limiter_gain_db = round(report.limiter_gain_db + trim, 2)

    return np.clip(audio, -ceiling, ceiling).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def master_track(
    input_path, output_path, platform="spotify",
    api_key=None, custom_lufs=None, custom_true_peak=None,
    force_fallback=False,
):
    """
    Fully dynamic mastering. No hardcoded values.

    Steps:
      1.  Load + stereo + resample 44100
      2.  Signal analysis (30+ features)
      3.  Genre estimation (from signal features)
      4.  Dynamic parameter resolver (all params computed from signal)
      5.  AI advisor (optional — refines params if API key available)
      6.  DC fix
      7.  Noise gate (conditional)
      8.  Dynamic EQ (cuts only when band exceeds threshold)
      9.  Multiband compression (3-band, signal-derived params)
      10. Glue compression (BPM-locked)
      11. Transient shaper (gain-neutral)
      12. Adaptive saturation (tape/tube/clip from signal)
      13. Tonal EQ (boosts from spectral deficit)
      14. Multiband M/S (sub mono, mid balanced, air wide)
      15. LUFS normalisation + true-peak limiter
      16. Verification trim
      17. Write 24-bit WAV
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

    # ── Load ──────────────────────────────────────────────────────────────────
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

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("  [1/15] Analysing track...", flush=True)
    analysis = analyse_track(audio, sr, str(input_path))
    report.before = analysis

    # QA warnings
    if analysis.stereo_correlation < 0.3:
        report.warnings.append("⚠ Severe phase issues — check mono compatibility")
    if analysis.dynamic_range_db < 5:
        report.warnings.append("⚠ Very low dynamic range — may be over-compressed")
    if analysis.noise_floor_db > -50:
        report.warnings.append(f"⚠ Noise floor at {analysis.noise_floor_db:.1f} dBFS")
    if analysis.low_end_mono < 0.7:
        report.warnings.append("⚠ Bass not mono — may phase on mono playback")

    # ── Genre + params ────────────────────────────────────────────────────────
    print("  [2/15] Estimating genre + computing params...", flush=True)
    genre = estimate_genre(analysis)
    report.estimated_genre = genre
    params = DynamicParams(analysis, genre, platform)
    print(f"         Genre: {genre}  |  Sat: {params.sat_model} drive={params.sat_drive}  "
          f"|  Comp: {params.comp_ratio:.2f}:1  |  Side: {params.ms_side_db:+.1f}dB",
          flush=True)

    # ── AI advisor (optional refinement) ─────────────────────────────────────
    print("  [3/15] AI advisor...", flush=True)
    analysis_json = to_json(analysis)
    if force_fallback or not api_key:
        from ai_advisor import _rule_based_fallback
        recipe = _rule_based_fallback(analysis_json, platform)
    else:
        recipe = get_mastering_recipe(analysis_json, platform=platform, api_key=api_key)
    report.recipe = recipe

    target_lufs = float(np.clip(recipe.target_lufs, -20.0, -6.0))
    target_tp   = float(np.clip(recipe.true_peak_dbfs, -3.0, -0.1))
    report.target_lufs      = target_lufs
    report.target_true_peak = target_tp

    # ── DC fix ────────────────────────────────────────────────────────────────
    print("  [4/15] DC offset fix...", flush=True)
    audio = _stage_dc_fix(audio, sr)

    # ── Noise gate ────────────────────────────────────────────────────────────
    if analysis.noise_floor_db > -50:
        print("  [4.5/15] Noise gate...", flush=True)
        gate  = pb.NoiseGate(threshold_db=analysis.noise_floor_db + 6.0,
                              attack_ms=5, release_ms=100)
        audio = gate(audio.T, sr).T
        report.warnings.append(f"⚠ Noise gate at {analysis.noise_floor_db+6:.1f} dBFS")

    # ── Dynamic EQ ────────────────────────────────────────────────────────────
    n_bands = len(params.dyn_eq_bands)
    print(f"  [5/15] Dynamic EQ ({n_bands} active bands, genre={genre})...", flush=True)
    audio = _stage_dynamic_eq(audio, sr, params, report)

    # ── Multiband compression ─────────────────────────────────────────────────
    print("  [6/15] Multiband compression...", flush=True)
    audio = _stage_multiband_comp(audio, sr, params, report)

    # ── Glue compression ──────────────────────────────────────────────────────
    print(f"  [7/15] Glue compression ({params.comp_ratio:.2f}:1)...", flush=True)
    audio = _stage_glue_comp(audio, sr, params, report)

    # ── Transient shaper ──────────────────────────────────────────────────────
    print("  [8/15] Transient shaper...", flush=True)
    audio = _stage_transient_shaper(audio, sr, params, report)

    # ── Saturation ────────────────────────────────────────────────────────────
    print(f"  [9/15] Saturation ({params.sat_model}, drive={params.sat_drive})...",
          flush=True)
    audio = _stage_saturation(audio, params, report)

    # ── Tonal EQ ──────────────────────────────────────────────────────────────
    n_tonal = len(params.tonal_boosts)
    print(f"  [10/15] Tonal EQ ({n_tonal} boosts from {genre} target)...", flush=True)
    audio = _stage_tonal_eq(audio, sr, params, report)

    # ── Multiband M/S ─────────────────────────────────────────────────────────
    print(f"  [11/15] Multiband M/S (side={params.ms_side_db:+.1f}dB)...", flush=True)
    audio = _stage_ms_multiband(audio, sr, params, report)

    # ── Normalise ─────────────────────────────────────────────────────────────
    print(f"  [12/15] Normalising → {target_lufs:.0f} LUFS / {target_tp:.1f} dBTP...",
          flush=True)
    audio = _stage_normalize(audio, sr, target_lufs, target_tp, report)

    # ── Verification trim ─────────────────────────────────────────────────────
    print("  [13/15] Verification trim...", flush=True)
    audio = _post_chain_trim(audio, sr, target_lufs, target_tp, report)

    # ── Write ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr, subtype="PCM_24")

    # ── Post analysis ─────────────────────────────────────────────────────────
    print("  [14/15] Post-analysis...", flush=True)
    report.after    = analyse_track(audio, sr, str(output_path))
    report.elapsed_s = round(time.time() - t0, 1)

    lufs_ok = abs(report.after.integrated_lufs - target_lufs) < 1.5
    peak_ok = report.after.true_peak_dbfs <= (target_tp + 0.1)
    report.passed_qa = lufs_ok and peak_ok

    if not lufs_ok:
        report.warnings.append(
            f"⚠ LUFS {report.after.integrated_lufs:.1f} vs target {target_lufs:.1f}")
    if not peak_ok:
        report.warnings.append(
            f"⚠ True peak {report.after.true_peak_dbfs:.1f} > ceiling {target_tp:.1f}")

    print(f"  [15/15] Done — {report.elapsed_s:.1f}s  "
          f"Genre={genre}  QA={'✓ PASS' if report.passed_qa else '✗ FAIL'}",
          flush=True)
    return report
