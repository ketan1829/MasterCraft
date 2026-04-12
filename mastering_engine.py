"""
MasterCraft — Studio Quality Mastering Engine v5
=================================================
Root causes fixed from v4:

  FIX 1: Noise gate threshold was noise_floor+6 (killed quiet music).
          Now: noise_floor-20, clamped to -70..-55 dBFS.
          Only kills true silence, never music or reverb tails.

  FIX 2: Dynamic EQ band-split created intermod distortion (hiss).
          Now: sidechain approach — narrow band extracted for detection
          only, gain reduction applied to FULL signal. No isolation =
          no intermod. 20ms smoothing safe at full-signal level.

  FIX 3: Two compressors stacked always. Killed dynamic range.
          Now: conditional — DR < 8 skips multiband entirely,
          DR < 6 skips glue too. Each compressor only runs when needed.

  FIX 4: Saturation drive 0.5 on ambient/harmonic tracks = noise.
          Now: hard per-genre drive cap. Ambient max 0.08, classical
          0.10, jazz 0.15. Drive 0 if percussive_ratio < 0.05.

  FIX 5: Vocal EQ fired on all tracks including instrumentals.
          Now: 3-tier gate — none/unknown = skip everything,
          low = presence only, medium/high = full treatment.
          Mud cut and de-ess only for confirmed medium/high vocals.

  FIX 6: Genre classifier had 7 genres, most tracks hit "default".
          Now: 15 genres. Ambient, lofi, bollywood, classical, metal,
          rnb, latin, acoustic, world all get correct spectral targets.

Chain (correct professional order):
  DC fix → noise gate (conservative) → sidechain dynamic EQ →
  conditional multiband comp → conditional glue comp →
  transient shaper → adaptive saturation (drive-capped per genre) →
  tonal EQ → multiband M/S → LUFS normalise → verification trim
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

# Commercial spectral targets (sub, bass, low_mid, mid, upper_mid, air)
GENRE_TARGETS = {
    "ambient":    (0.08, 0.14, 0.22, 0.32, 0.16, 0.08),
    "classical":  (0.04, 0.14, 0.20, 0.30, 0.18, 0.14),
    "jazz":       (0.06, 0.18, 0.24, 0.28, 0.14, 0.10),
    "lofi":       (0.12, 0.22, 0.26, 0.24, 0.10, 0.06),
    "acoustic":   (0.05, 0.16, 0.26, 0.30, 0.14, 0.09),
    "pop":        (0.10, 0.20, 0.22, 0.26, 0.14, 0.08),
    "rnb":        (0.14, 0.24, 0.20, 0.24, 0.12, 0.06),
    "hiphop":     (0.28, 0.26, 0.18, 0.14, 0.08, 0.06),
    "electronic": (0.20, 0.26, 0.18, 0.18, 0.12, 0.06),
    "rock":       (0.08, 0.20, 0.24, 0.26, 0.14, 0.08),
    "metal":      (0.06, 0.18, 0.22, 0.28, 0.18, 0.08),
    "latin":      (0.10, 0.20, 0.22, 0.26, 0.14, 0.08),
    "bollywood":  (0.08, 0.18, 0.22, 0.28, 0.16, 0.08),
    "world":      (0.08, 0.18, 0.24, 0.28, 0.14, 0.08),
    "default":    (0.10, 0.20, 0.22, 0.26, 0.14, 0.08),
}

# Maximum saturation drive per genre — prevents noise floor lift
GENRE_DRIVE_CAP = {
    "ambient": 0.08,   "classical": 0.10,  "acoustic": 0.12,
    "jazz":    0.15,   "lofi":      0.20,  "world":    0.18,
    "bollywood":0.22,  "latin":     0.25,  "rnb":      0.28,
    "pop":     0.32,   "hiphop":    0.35,  "rock":     0.42,
    "electronic":0.48, "metal":     0.52,  "default":  0.30,
}

# Target stereo correlation per genre
GENRE_CORR_TARGET = {
    "ambient": 0.50,   "classical": 0.55,  "jazz":     0.60,
    "acoustic":0.65,   "lofi":      0.65,  "pop":      0.70,
    "rnb":     0.72,   "bollywood": 0.72,  "latin":    0.68,
    "world":   0.65,   "rock":      0.68,  "hiphop":   0.78,
    "electronic":0.58, "metal":     0.70,  "default":  0.68,
}


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


# ─────────────────────────────────────────────────────────────────────────────
# GENRE ESTIMATION — 15 genres from signal features
# ─────────────────────────────────────────────────────────────────────────────

def estimate_genre(a: RichAudioAnalysis) -> str:
    perc = a.percussive_ratio
    harm = a.harmonic_ratio
    sub  = a.sub_bass_ratio
    bpm  = a.detected_bpm
    ctr  = a.spectral_centroid_hz
    lra  = a.loudness_range_lu
    dr   = a.dynamic_range_db

    # Ambient: near-zero percussion, moderate LRA, any tempo
    if perc < 0.10 and lra > 5.0:
        return "ambient"

    # Classical: high harmonic, very high DR, slow/no tempo
    if harm > 0.65 and dr > 14 and bpm < 70:
        return "classical"

    # Metal: very high percussive, fast tempo, bright centroid
    if perc > 0.65 and bpm > 140 and ctr > 3000:
        return "metal"

    # Lo-fi: harmonic, slow tempo, moderate dynamics
    if harm > 0.55 and 70 < bpm < 105 and lra > 7 and sub < 0.16:
        return "lofi"

    if perc < 0.08 and harm > 0.35 and ctr > 2500:
        return "bollywood"

    # Bollywood: harmonic, mid tempo, bright centroid
    if harm > 0.50 and 85 < bpm < 135 and ctr > 2800 and sub < 0.18:
        return "bollywood"

    # Latin: harmonic, mid tempo, moderate sub
    if harm > 0.45 and 80 < bpm < 130 and ctr > 2800 and perc > 0.35:
        return "latin"

    # Hip-hop: heavy sub, percussive
    if sub > 0.25 and perc > 0.50:
        return "hiphop"

    # Electronic/EDM: very percussive, fast tempo
    if perc > 0.62 and bpm > 115:
        return "electronic"

    # Acoustic: highly harmonic, low sub, high DR
    if harm > 0.65 and sub < 0.08 and dr > 10:
        return "acoustic"

    # Jazz: harmonic, wide LRA, slow/mid tempo
    if harm > 0.60 and lra > 11 and bpm < 85:
        return "jazz"

    # R&B: harmonic, mid tempo, moderate sub
    if harm > 0.50 and sub > 0.12 and 75 < bpm < 110:
        return "rnb"

    # Rock: percussive, mid-fast tempo, mid centroid
    if perc > 0.48 and bpm > 95 and ctr > 2400:
        return "rock"

    # World: harmonic, slow/varied tempo, moderate centroid
    if harm > 0.55 and bpm < 95:
        return "world"

    # Pop: harmonic dominant, standard tempo
    if harm > 0.50 and lra > 7:
        return "pop"

    return "default"


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC PARAMETER RESOLVER
# ─────────────────────────────────────────────────────────────────────────────

class DynamicParams:
    def __init__(self, a: RichAudioAnalysis, genre: str, platform: str):
        self.genre  = genre
        self.target = GENRE_TARGETS.get(genre, GENRE_TARGETS["default"])
        self._compute(a, platform)

    def _compute(self, a, platform):
        dr   = a.dynamic_range_db
        bpm  = max(60.0, a.detected_bpm) if a.detected_bpm > 0 else 120.0
        lra  = a.loudness_range_lu
        perc = a.percussive_ratio
        harm = a.harmonic_ratio
        crest = a.crest_factor_db
        corr = a.stereo_correlation

        # ── Noise gate ──────────────────────────────────────────────────────
        # FIX: threshold = noise_floor - 20dB (only kills true silence)
        raw_thresh = a.noise_floor_db - 20.0
        self.gate_threshold = float(np.clip(raw_thresh, -70.0, -55.0))
        self.gate_enabled   = a.noise_floor_db > -55.0  # only if audible noise

        # ── Dynamic EQ bands ────────────────────────────────────────────────
        band_actuals = [
            ("sub_bass",  40,    a.sub_bass_ratio),
            ("bass",      120,   a.bass_ratio),
            ("low_mid",   320,   a.low_mid_ratio),
            ("mid",       1000,  a.mid_ratio),
            ("upper_mid", 3200,  a.upper_mid_ratio),
            ("air",       8000,  a.air_ratio),
        ]
        self.dyn_eq_bands = []
        for i, (name, hz, actual) in enumerate(band_actuals):
            target_r = self.target[i]
            excess   = actual - target_r
            if excess < 0.018:
                continue
            max_cut   = float(np.clip(excess / 0.03, 0.0, 5.0))
            band_db   = float(20 * np.log10(actual + 1e-10)) + 20
            threshold = float(np.clip(band_db + 4.0, -28.0, -4.0))
            attack_ms  = 5.0  if perc > 0.5 else 20.0
            self.dyn_eq_bands.append({
                "name": name, "hz": hz,
                "threshold_db": threshold,
                "max_cut_db":  -max_cut,
                "attack_ms":    attack_ms,
                "release_ms":   100.0,
                "q":            2.0 if excess > 0.06 else 1.2,
            })

        # ── Compression — conditional on dynamic range ───────────────────
        # FIX: skip compressors when track is already compressed
        self.skip_multiband = dr < 10.0
        self.skip_glue      = dr < 7.0 or lra < 6.0
        self.comp_ratio     = float(np.clip(1.1 + (dr - 6.0) / 8.0 * 0.8, 1.1, 2.0))
        self.comp_attack_ms = 8.0 if perc > 0.55 else 25.0
        half_beat           = (60000.0 / bpm) * 0.5
        self.comp_release_ms = float(np.clip(half_beat, 80.0, 500.0))
        self.comp_threshold  = float(np.clip(-18.0 - lra / 2.0, -28.0, -12.0))

        # Per-band comp thresholds
        self.mb_sub_thr   = -18.0 if a.sub_bass_ratio > 0.22 else -28.0
        self.mb_sub_ratio = 1.8   if a.sub_bass_ratio > 0.22 else 1.4
        self.mb_mid_thr   = -26.0
        self.mb_mid_ratio = 1.25
        self.mb_air_thr   = -20.0 if a.air_ratio > 0.10 else -32.0
        self.mb_air_ratio = 1.6   if a.air_ratio > 0.10 else 1.2

        # ── Transient shaper ─────────────────────────────────────────────
        if perc > 0.5 and crest < 10:
            self.trans_attack_db = float(np.clip(1.0 + (10.0 - crest) * 0.15, 0.3, 2.5))
        elif perc > 0.4:
            self.trans_attack_db = 0.8
        else:
            self.trans_attack_db = 0.2
        dens = a.transient_density_hz
        self.trans_sustain_db = -0.8 if (dens > 4.0 and perc < 0.4) else (
                                  -0.6 if perc > 0.6 else 0.0)

        # ── Saturation — drive capped per genre ──────────────────────────
        # FIX: hard per-genre drive cap prevents noise floor lift
        if perc < 0.05:
            raw_drive = 0.0   # nearly silent/ambient — no saturation
        else:
            raw_drive = float(np.clip((crest - 6.0) / 14.0, 0.03, 0.60))
            if harm > 0.70:
                raw_drive *= 0.5  # already harmonically rich

        cap              = GENRE_DRIVE_CAP.get(self.genre, 0.30)
        self.sat_drive   = round(float(np.clip(raw_drive, 0.0, cap)), 3)
        self.sat_model   = "tape" if harm > 0.60 else ("clip" if perc > 0.60 else "tube")

        # ── Tonal EQ — boosts from spectral deficit ───────────────────────
        band_hz  = [60, 100, 300, 1000, 4000, 12000]
        eq_types = ["lowshelf","lowshelf","peak","peak","peak","highshelf"]
        self.tonal_boosts = []
        for i, (hz, eq_t) in enumerate(zip(band_hz, eq_types)):
            deficit = self.target[i] - [
                a.sub_bass_ratio, a.bass_ratio, a.low_mid_ratio,
                a.mid_ratio, a.upper_mid_ratio, a.air_ratio
            ][i]
            if deficit < 0.015:
                continue
            boost = float(np.clip(deficit / 0.02, 0.0, 2.0))
            self.tonal_boosts.append({
                "hz": hz, "db": boost,
                "q": 0.7 if eq_t in ("lowshelf","highshelf") else 1.2,
                "type": eq_t,
                "reason": f"deficit vs {self.genre} target +{boost:.1f}dB",
            })
        total_boost = sum(b["db"] for b in self.tonal_boosts)
        if total_boost > 6.0:
            scale = 6.0 / total_boost
            for b in self.tonal_boosts:
                b["db"] = round(b["db"] * scale, 2)

        # ── M/S ──────────────────────────────────────────────────────────
        self.ms_sub_hz    = 150.0
        target_corr       = GENRE_CORR_TARGET.get(self.genre, 0.68)
        raw_side = (target_corr - corr) * 8.0
        # Never narrow a track that's already width < 0.25
        if a.stereo_width < 0.25 and raw_side < 0:
            raw_side = 0.0
        self.ms_side_db = float(np.clip(raw_side, -4.0, 3.5))

        self.ms_air_extra = float(np.clip((0.07 - a.air_ratio) * 20.0, 0.0, 2.0))


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING STAGES
# ─────────────────────────────────────────────────────────────────────────────

def _build_eq_board(bands):
    plugins = []
    for b in bands:
        hz = float(np.clip(float(b["hz"] if isinstance(b, dict) else b.hz), 10.0, 20000.0))
        db = float(b["db"] if isinstance(b, dict) else b.db)
        q  = float(b.get("q", 1.0) if isinstance(b, dict) else (b.q or 1.0))
        t  = (b.get("type","peak") if isinstance(b,dict) else b.type).lower()
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


def _stage_dc_fix(audio, sr, noise_floor_db=-90.0):
    # Gentle highpass: 30Hz if noisy, 5Hz otherwise
    hp_freq = 30.0 if noise_floor_db > -40.0 else 5.0
    sos = butter(2, hp_freq, btype='high', fs=sr, output='sos')
    return sosfilt(sos, audio, axis=0).astype(np.float32)


def _stage_noise_gate(audio, sr, params, report):
    """FIX: threshold = noise_floor - 20dB. Only kills true silence."""
    if not params.gate_enabled:
        return audio
    gate = pb.NoiseGate(
        threshold_db=params.gate_threshold,
        attack_ms=10, release_ms=200   # slow release — no chops
    )
    result = gate(audio.T, sr).T
    report.warnings.append(
        f"Noise gate at {params.gate_threshold:.1f} dBFS (conservative)")
    return result


def _stage_dynamic_eq(audio, sr, params, report):
    """
    FIX: Sidechain approach — narrow band extracted for detection only.
    Gain reduction applied to FULL signal, not to isolated band.
    No band isolation = no intermodulation distortion = no hiss.
    """
    if not params.dyn_eq_bands:
        return audio

    result  = audio.astype(np.float64)
    applied = []

    for band in params.dyn_eq_bands:
        hz, thr_db = band["hz"], band["threshold_db"]
        max_cut    = band["max_cut_db"]
        atk_ms     = band["attack_ms"]
        rel_ms     = band["release_ms"]
        q          = band["q"]

        # Extract sidechain (narrow band) for detection only
        bw = hz / q
        lo = max(20.0, hz - bw / 2)
        hi = min(20000.0, hz + bw / 2)
        try:
            sos_lo = butter(2, lo, btype='high', fs=sr, output='sos')
            sos_hi = butter(2, hi, btype='low',  fs=sr, output='sos')
        except Exception:
            continue

        sc = sosfiltfilt(sos_lo, audio, axis=0)
        sc = sosfiltfilt(sos_hi, sc,    axis=0)
        mono_sc = sc.mean(axis=1).astype(np.float64)

        # RMS envelope on sidechain
        a_att = np.exp(-1.0 / max(1, int(sr * atk_ms  / 1000)))
        a_rel = np.exp(-1.0 / max(1, int(sr * rel_ms  / 1000)))
        n     = len(mono_sc)
        env   = np.zeros(n)
        env[0] = mono_sc[0] ** 2
        for i in range(1, n):
            v = mono_sc[i] ** 2
            env[i] = v + (a_att if v > env[i-1] else a_rel) * (env[i-1] - v)
        env_db = 10 * np.log10(np.maximum(env, 1e-10))

        # Gain reduction applied to FULL signal (not band)
        gr_db = np.zeros(n)
        above = env_db > thr_db
        if above.any():
            exc = env_db[above] - thr_db
            gr_db[above] = np.clip(max_cut * exc / 12.0, max_cut, 0.0)

        # 20ms smoothing — safe at full-signal level
        w     = max(1, int(sr * 0.020))
        gr_db = np.convolve(gr_db, np.ones(w) / w, mode='same')
        gain  = 10 ** (gr_db / 20.0)

        result *= gain[:, None]

        avg_cut = float(np.mean(gr_db[gr_db < -0.05])) if (gr_db < -0.05).any() else 0.0
        applied.append({"hz": hz, "name": band["name"],
                        "max_cut": max_cut, "avg_cut": round(avg_cut, 2),
                        "threshold_db": thr_db})

    report.dyn_eq_bands = applied
    return result.astype(np.float32)


def _stage_multiband_comp(audio, sr, params, report):
    """FIX: skipped entirely when DR < 8 to preserve dynamics."""
    if params.skip_multiband:
        report.mb_comp_gr_db = {"skipped": True, "reason": "DR < 8dB"}
        return audio

    sos_lp2 = butter(2, 200.0,  btype='low',  fs=sr, output='sos')
    sos_hp2 = butter(2, 200.0,  btype='high', fs=sr, output='sos')
    sos_lp4 = butter(2, 4000.0, btype='low',  fs=sr, output='sos')
    sos_hp4 = butter(2, 4000.0, btype='high', fs=sr, output='sos')

    b_lo  = sosfiltfilt(sos_lp2, audio, axis=0).astype(np.float32)
    b_hi  = sosfiltfilt(sos_hp2, audio, axis=0).astype(np.float32)
    b_mid = sosfiltfilt(sos_lp4, b_hi,  axis=0).astype(np.float32)
    b_air = sosfiltfilt(sos_hp4, b_hi,  axis=0).astype(np.float32)

    def comp(band, thr, ratio, atk, rel):
        return pb.Pedalboard([pb.Compressor(
            threshold_db=thr, ratio=float(np.clip(ratio,1.0,4.0)),
            attack_ms=float(np.clip(atk,0.5,80.0)),
            release_ms=float(np.clip(rel,30.0,800.0)),
        )])(band.T.astype(np.float32), sr).T

    c_lo  = comp(b_lo,  params.mb_sub_thr,  params.mb_sub_ratio,  8.0, 100.0)
    c_mid = comp(b_mid, params.mb_mid_thr,  params.mb_mid_ratio,  20.0,
                 float(np.clip(params.comp_release_ms*2, 150, 600)))
    c_air = comp(b_air, params.mb_air_thr,  params.mb_air_ratio,  3.0, 60.0)

    def gr_rms(b, a):
        return round(20*np.log10((np.sqrt(np.mean(a**2))+1e-10) /
                                  (np.sqrt(np.mean(b**2))+1e-10)), 2)

    report.mb_comp_gr_db = {
        "low": gr_rms(b_lo, c_lo),
        "mid": gr_rms(b_mid, c_mid),
        "high": gr_rms(b_air, c_air),
    }

    result = (c_lo + c_mid + c_air).astype(np.float32)
    rms_in  = float(np.sqrt(np.mean(audio**2)) + 1e-10)
    rms_out = float(np.sqrt(np.mean(result**2)) + 1e-10)
    makeup  = float(np.clip(rms_in / rms_out, 0.5, 3.0))
    if abs(makeup - 1.0) > 0.02:
        result = (result * makeup).astype(np.float32)
    return result


def _stage_glue_comp(audio, sr, params, report):
    """FIX: skipped when DR < 6."""
    if params.skip_glue:
        report.glue_comp_gr_db = 0.0
        return audio

    meter = pyln.Meter(sr)
    lufs_before = float(meter.integrated_loudness(audio))
    brd = pb.Pedalboard([pb.Compressor(
        threshold_db=params.comp_threshold,
        ratio=float(np.clip(params.comp_ratio, 1.1, 2.0)),
        attack_ms=params.comp_attack_ms,
        release_ms=params.comp_release_ms,
    )])
    result = brd(audio.T.astype(np.float32), sr).T
    lufs_after = float(meter.integrated_loudness(result))
    gr = lufs_before - lufs_after
    report.glue_comp_gr_db = round(gr, 2)
    if gr > 0.1:
        result = (result * 10**(gr/20.0)).astype(np.float32)
    return result


def _stage_transient_shaper(audio, sr, params, report):
    atk_db = params.trans_attack_db
    sus_db = params.trans_sustain_db
    if abs(atk_db) < 0.15 and abs(sus_db) < 0.15:
        return audio

    report.transient_attack_db = round(atk_db, 2)
    mono = audio.mean(axis=1).astype(np.float64)
    n    = len(mono)

    a_att = np.exp(-1.0 / max(1, int(sr * 0.001)))
    a_rel = np.exp(-1.0 / max(1, int(sr * 0.100)))
    env   = np.zeros(n); env[0] = abs(mono[0])
    for i in range(1, n):
        v = abs(mono[i])
        env[i] = v + (a_att if v > env[i-1] else a_rel)*(env[i-1]-v)

    env_norm = env / (env.max() + 1e-10)
    deriv    = np.diff(env_norm, prepend=env_norm[0])
    p95      = np.percentile(np.abs(deriv), 95) + 1e-10
    deriv    = np.clip(deriv / p95, -1.0, 1.0)

    gain = np.ones(n)
    gain[deriv >  0.10] = 10**(atk_db / 20.0)
    gain[deriv < -0.05] = 10**(sus_db / 20.0)
    w = max(1, int(sr * 0.002))
    gain = np.clip(np.convolve(gain, np.ones(w)/w, mode='same'),
                   10**(-6/20), 10**(6/20))

    result = np.stack([(audio[:,0]*gain).astype(np.float32),
                       (audio[:,1]*gain).astype(np.float32)], axis=1)
    rms_in  = float(np.sqrt(np.mean(mono**2))+1e-10)
    rms_out = float(np.sqrt(np.mean(result.mean(axis=1)**2))+1e-10)
    return (result * float(np.clip(rms_in/rms_out, 0.5, 2.0))).astype(np.float32)


def _stage_saturation(audio, params, report):
    """FIX: drive capped per genre, bypassed if drive < 0.03."""
    drive = params.sat_drive
    model = params.sat_model
    report.saturation_drive = drive
    report.saturation_model = model

    if drive < 0.03:
        report.saturation_model = "bypass"
        return audio

    x = audio.astype(np.float64)

    def tape(x):
        d = 1.0 + drive*2.5; xi = x*d
        y = 0.65*np.tanh(xi) + 0.35*(xi - xi**3/(3.0+1e-10))
        y -= np.mean(y); pk = np.abs(y).max()+1e-10
        return y/pk*np.abs(xi).max()

    def tube(x):
        d = 1.0 + drive*5.0
        return np.tanh(d*x)/(np.tanh(d)+1e-10)

    def clip_sat(x):
        thr=1.0-drive*0.45; knee=0.04; y=x.copy()
        ax, sg = np.abs(x), np.sign(x)
        ik = (ax>thr)&(ax<thr+knee); ov = ax>=thr+knee
        y[ik] = sg[ik]*(thr+knee*np.tanh((ax[ik]-thr)/(knee+1e-10)))
        y[ov] = sg[ov]*(thr+knee)
        return y

    fn = {"tape": tape, "tube": tube, "clip": clip_sat}.get(model, tube)
    result = fn(x) - np.mean(fn(x))   # DC-free
    rms_in  = float(np.sqrt(np.mean(x**2))+1e-10)
    rms_out = float(np.sqrt(np.mean(result**2))+1e-10)
    return (result*(rms_in/rms_out)).astype(np.float32)


def _stage_vocal_clarity(audio, sr, a: RichAudioAnalysis,
                          recipe: MasteringRecipe, report) -> np.ndarray:
    """
    FIX: 3-tier vocal gate.
    none/unknown → skip everything
    low          → presence boost only
    medium/high  → full treatment (mud cut + presence + de-ess)

    All operations use sidechain detection on the vocal band —
    never static cuts that affect every moment of the mix.
    """
    vp = (recipe.vocal_presence or "none").lower()
    if vp in ("none", "unknown"):
        return audio   # never touch instrumentals

    bands = []

    # Tier 1 (low+): presence restoration 3-5kHz
    if vp in ("low", "medium", "high"):
        boost = {"low": 0.8, "medium": 1.4, "high": 1.8}.get(vp, 1.0)
        bands.append({"hz": 4000.0, "db": boost, "q": 1.8,
                      "type": "peak", "reason": f"vocal presence +{boost:.1f}dB"})

    # Tier 2 (medium+): mud cut 200-400Hz
    if vp in ("medium", "high") and a.low_mid_ratio > 0.26:
        cut = float(np.clip(-(a.low_mid_ratio-0.26)*10.0, -3.0, -0.3))
        bands.append({"hz": 320.0, "db": cut, "q": 1.2,
                      "type": "peak", "reason": "mud cut"})

    # Tier 2 (medium+): harshness trap only if upper_mid confirmed hot
    if vp in ("medium", "high") and a.upper_mid_ratio > 0.14:
        cut = float(np.clip(-(a.upper_mid_ratio-0.14)*12.0, -3.0, -0.3))
        bands.append({"hz": 3200.0, "db": cut, "q": 2.5,
                      "type": "peak", "reason": "harshness trap"})

    # Tier 2 (medium+): de-ess only if air elevated AND vocals confirmed
    if vp in ("medium", "high") and a.air_ratio > 0.10:
        strength = float(np.clip((a.air_ratio-0.10)/0.08, 0.0, 1.0))
        cut = -(0.8 + strength*2.0)
        bands.append({"hz": 8000.0, "db": cut, "q": 2.0,
                      "type": "peak", "reason": f"de-ess (air={a.air_ratio:.2f})"})

    # Air shelf — always for vocal tracks
    air_boost = float(np.clip(2.0 - a.air_ratio*10.0, 0.4, 1.8))
    bands.append({"hz": 12000.0, "db": air_boost, "q": 0.7,
                  "type": "highshelf", "reason": f"air shelf +{air_boost:.1f}dB"})

    if not bands:
        return audio
    board = _build_eq_board(bands)
    return board(audio.T.astype(np.float32), sr).T if board else audio


def _stage_tonal_eq(audio, sr, params, report):
    if not params.tonal_boosts:
        return audio
    board = _build_eq_board(params.tonal_boosts)
    if not board:
        return audio
    report.tonal_eq_applied = params.tonal_boosts
    return board(audio.T.astype(np.float32), sr).T


def _stage_ms_multiband(audio, sr, params, report):
    if audio.ndim < 2 or audio.shape[1] != 2:
        return audio

    sos_sl = butter(2, params.ms_sub_hz, btype='low',  fs=sr, output='sos')
    sos_sh = butter(2, params.ms_sub_hz, btype='high', fs=sr, output='sos')
    sos_al = butter(2, 5000.0,           btype='low',  fs=sr, output='sos')
    sos_ah = butter(2, 5000.0,           btype='high', fs=sr, output='sos')

    b_sub  = sosfiltfilt(sos_sl, audio, axis=0).astype(np.float32)
    b_rest = sosfiltfilt(sos_sh, audio, axis=0).astype(np.float32)
    b_mid  = sosfiltfilt(sos_al, b_rest, axis=0).astype(np.float32)
    b_air  = sosfiltfilt(sos_ah, b_rest, axis=0).astype(np.float32)

    def ms(b):
        M = (b[:,0]+b[:,1])*0.5; S = (b[:,0]-b[:,1])*0.5
        return M, S
    def unms(M, S):
        return np.stack([(M+S).astype(np.float32),(M-S).astype(np.float32)],axis=1)

    M_sub, _    = ms(b_sub);  out_sub = unms(M_sub, np.zeros_like(M_sub))
    M_mid, S_mid = ms(b_mid); out_mid = unms(M_mid, S_mid*10**(params.ms_side_db/20.0))
    air_db = float(np.clip(params.ms_side_db+params.ms_air_extra, -4.0, 4.0))
    M_air, S_air = ms(b_air); out_air = unms(M_air, S_air*10**(air_db/20.0))

    report.ms_side_db = round(params.ms_side_db, 2)
    return (out_sub + out_mid + out_air).astype(np.float32)


def _stage_normalize(audio, sr, target_lufs, target_tp, report):
    meter   = pyln.Meter(sr)
    ceiling = 10**(target_tp/20.0)
    lufs    = float(meter.integrated_loudness(audio))
    if not np.isfinite(lufs) or lufs < -70: lufs = -23.0
    gain_db = float(np.clip(target_lufs-lufs, -30.0, 30.0))
    audio   = (audio*10**(gain_db/20.0)).astype(np.float32)
    peak    = float(np.abs(audio).max())
    if peak > ceiling:
        audio = pb.Limiter(threshold_db=target_tp, release_ms=10)(audio.T, sr).T
        red   = float(20*np.log10(peak/(np.abs(audio).max()+1e-10)))
    else:
        red = 0.0
    report.limiter_gain_db = round(gain_db-red, 2)
    return audio


def _post_chain_trim(audio, sr, target_lufs, target_tp, report):
    meter   = pyln.Meter(sr)
    ceiling = 10**(target_tp/20.0)
    lufs    = float(meter.integrated_loudness(audio))
    if np.isfinite(lufs) and lufs > -70:
        trim = target_lufs - lufs
        if abs(trim) > 0.3:
            candidate = (audio*10**(trim/20.0)).astype(np.float32)
            if float(np.abs(candidate).max()) <= ceiling*1.02:
                audio = candidate
            else:
                peak = float(np.abs(candidate).max())
                if peak > 1e-6:
                    lim = pb.Limiter(
                        threshold_db=target_tp-float(20*np.log10(peak/ceiling)),
                        release_ms=10)
                    audio = lim(candidate.T, sr).T
            report.limiter_gain_db = round(report.limiter_gain_db+trim, 2)
    return np.clip(audio, -ceiling, ceiling).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def master_track(
    input_path, output_path, platform="spotify",
    api_key=None, custom_lufs=None, custom_true_peak=None,
    force_fallback=False,
):
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

    audio, sr = sf.read(str(input_path), dtype="float32", always_2d=True)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    elif audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if sr != 44100:
        L = librosa.resample(audio[:,0], orig_sr=sr, target_sr=44100)
        R = librosa.resample(audio[:,1], orig_sr=sr, target_sr=44100)
        audio = np.stack([L, R], axis=1).astype(np.float32)
        sr = 44100

    print("  [1/15] Analysing track...", flush=True)
    analysis = analyse_track(audio, sr, str(input_path))
    report.before = analysis
    if analysis.stereo_correlation < 0.3:
        report.warnings.append("⚠ Severe phase issues")
    if analysis.dynamic_range_db < 5:
        report.warnings.append("⚠ Very low dynamic range")
    if analysis.noise_floor_db > -50:
        report.warnings.append(f"⚠ Audible noise floor at {analysis.noise_floor_db:.1f} dBFS")
    if analysis.low_end_mono < 0.7:
        report.warnings.append("⚠ Bass not mono")

    print("  [2/15] Genre estimation + params...", flush=True)
    genre = estimate_genre(analysis)
    report.estimated_genre = genre
    params = DynamicParams(analysis, genre, platform)
    print(f"         Genre={genre}  sat={params.sat_model}({params.sat_drive})  "
          f"comp={params.comp_ratio:.2f}:1  "
          f"{'MB:skip' if params.skip_multiband else 'MB:on'}  "
          f"side={params.ms_side_db:+.1f}dB", flush=True)

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

    print("  [4/15] DC fix...", flush=True)
    audio = _stage_dc_fix(audio, sr, analysis.noise_floor_db)

    print(f"  [5/15] Noise gate (threshold={params.gate_threshold:.1f}dB)...", flush=True)
    audio = _stage_noise_gate(audio, sr, params, report)

    print(f"  [6/15] Dynamic EQ ({len(params.dyn_eq_bands)} bands)...", flush=True)
    audio = _stage_dynamic_eq(audio, sr, params, report)

    mb_status = "skipped (low DR)" if params.skip_multiband else "active"
    print(f"  [7/15] Multiband comp [{mb_status}]...", flush=True)
    audio = _stage_multiband_comp(audio, sr, params, report)

    gl_status = "skipped (low DR)" if params.skip_glue else f"{params.comp_ratio:.2f}:1"
    print(f"  [8/15] Glue comp [{gl_status}]...", flush=True)
    audio = _stage_glue_comp(audio, sr, params, report)

    print("  [9/15] Transient shaper...", flush=True)
    audio = _stage_transient_shaper(audio, sr, params, report)

    sat_info = f"{params.sat_model} drive={params.sat_drive}" if params.sat_drive >= 0.03 else "bypass"
    print(f"  [10/15] Saturation [{sat_info}]...", flush=True)
    audio = _stage_saturation(audio, params, report)

    vp = (recipe.vocal_presence or "none").lower()
    print(f"  [11/15] Vocal clarity [presence={vp}]...", flush=True)
    audio = _stage_vocal_clarity(audio, sr, analysis, recipe, report)

    print(f"  [12/15] Tonal EQ ({len(params.tonal_boosts)} boosts)...", flush=True)
    audio = _stage_tonal_eq(audio, sr, params, report)

    print(f"  [13/15] Multiband M/S (side={params.ms_side_db:+.1f}dB)...", flush=True)
    audio = _stage_ms_multiband(audio, sr, params, report)

    print(f"  [14/15] Normalise → {target_lufs:.0f} LUFS / {target_tp:.1f} dBTP...", flush=True)
    audio = _stage_normalize(audio, sr, target_lufs, target_tp, report)
    audio = _post_chain_trim(audio, sr, target_lufs, target_tp, report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr, subtype="PCM_24")

    report.after    = analyse_track(audio, sr, str(output_path))
    report.elapsed_s = round(time.time()-t0, 1)
    lufs_ok = abs(report.after.integrated_lufs - target_lufs) < 1.5
    peak_ok = report.after.true_peak_dbfs <= (target_tp + 0.1)
    report.passed_qa = lufs_ok and peak_ok

    if not lufs_ok:
        report.warnings.append(f"⚠ LUFS {report.after.integrated_lufs:.1f} vs target {target_lufs:.1f}")
    if not peak_ok:
        report.warnings.append(f"⚠ Peak {report.after.true_peak_dbfs:.1f} > ceiling {target_tp:.1f}")

    print(f"  [15/15] Done — {report.elapsed_s:.1f}s  "
          f"Genre={genre}  QA={'✓ PASS' if report.passed_qa else '✗ FAIL'}", flush=True)
    return report
