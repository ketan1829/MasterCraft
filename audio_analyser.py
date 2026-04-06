"""
KetMaster — Rich Audio Analyser
================================
Extracts every meaningful signal descriptor from a track using only
librosa + scipy (no new installs). Produces a structured dict that
the AI advisor uses to make mastering decisions.

Features extracted:
  - Loudness: LUFS, true peak, LRA, RMS, crest factor
  - Dynamics: dynamic range, transient density, attack profile
  - Spectral: 31-band 1/3-octave ISO spectrum, centroid, rolloff,
              flatness, contrast, 13 MFCCs (timbre fingerprint)
  - Rhythm: BPM, beat regularity, onset density
  - Harmony: key, mode (major/minor), tonnetz, chord energy
  - Texture: harmonic/percussive ratio, zero crossing rate
  - Stereo: L/R correlation, stereo width estimate
  - Structure: section count estimate, silence ratio, noise floor
"""

from __future__ import annotations
import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import json

warnings.filterwarnings("ignore")

# ── ISO 1/3-octave band centre frequencies (Hz) ───────────────────────────────
ISO_BANDS_HZ = [
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150,
    4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000,
]

# ── Named frequency zones for mastering decisions ─────────────────────────────
FREQ_ZONES = {
    "sub_bass":     (20,   80),    # kick body, 808, rumble
    "bass":         (80,   200),   # bass fundamentals
    "low_mid":      (200,  500),   # mud zone, warmth
    "mid":          (500,  2000),  # vocals, guitar body
    "upper_mid":    (2000, 5000),  # presence, harshness, clarity
    "air":          (5000, 20000), # brilliance, sibilance, sparkle
}


@dataclass
class RichAudioAnalysis:
    """Complete audio descriptor — everything the AI advisor needs."""

    # ── Identity ──────────────────────────────────────────────────────────────
    file_path:          str   = ""
    duration_seconds:   float = 0.0
    sample_rate:        int   = 44100
    num_channels:       int   = 2

    # ── Loudness ──────────────────────────────────────────────────────────────
    integrated_lufs:    float = 0.0   # overall perceived loudness
    loudness_range_lu:  float = 0.0   # LRA — how wide the dynamics are
    true_peak_dbfs:     float = 0.0   # loudest intersample peak
    rms_db:             float = 0.0
    crest_factor_db:    float = 0.0   # peak/RMS — transient headroom

    # ── Dynamics ──────────────────────────────────────────────────────────────
    dynamic_range_db:      float = 0.0   # loud 10% vs quiet 50% of frames
    transient_density_hz:  float = 0.0   # onsets per second
    percussive_ratio:      float = 0.0   # 0=all-harmonic, 1=all-drums
    harmonic_ratio:        float = 0.0
    attack_rms_db:         float = 0.0   # avg RMS of first 10ms after onsets
    noise_floor_db:        float = -90.0 # quietest 5% of frames

    # ── Spectral (macro zones — ratios sum to 1.0) ────────────────────────────
    sub_bass_ratio:    float = 0.0
    bass_ratio:        float = 0.0
    low_mid_ratio:     float = 0.0
    mid_ratio:         float = 0.0
    upper_mid_ratio:   float = 0.0
    air_ratio:         float = 0.0

    # ── Spectral (detail) ─────────────────────────────────────────────────────
    spectral_centroid_hz:  float = 0.0   # "brightness centre of mass"
    spectral_rolloff_hz:   float = 0.0   # freq below which 85% of energy lives
    spectral_flatness:     float = 0.0   # 0=pure tone, 1=noise — "texture"
    spectral_contrast_mean: float = 0.0  # peaks vs valleys — "clarity"

    # 13 MFCCs as timbre fingerprint (mean values)
    mfcc_means: list[float] = field(default_factory=lambda: [0.0] * 13)

    # 31-band 1/3-octave spectrum as % of total energy
    iso_band_energies: list[float] = field(default_factory=lambda: [0.0] * 31)

    # ── Rhythm ────────────────────────────────────────────────────────────────
    detected_bpm:          float = 0.0
    beat_regularity:       float = 0.0   # 0=erratic, 1=metronomic
    zero_crossing_rate:    float = 0.0   # correlates with noisiness/texture

    # ── Harmony ───────────────────────────────────────────────────────────────
    key_estimate:    str   = "unknown"
    mode:            str   = "unknown"   # "major" or "minor"
    key_confidence:  float = 0.0         # 0–1 strength of key detection
    chord_complexity: float = 0.0        # avg chroma entropy — simple=low

    # ── Stereo ────────────────────────────────────────────────────────────────
    stereo_correlation: float = 1.0   # 1=mono, 0=wide, <0=phase issues
    stereo_width:       float = 0.0   # RMS of side channel relative to mid
    low_end_mono:       float = 1.0   # how mono the sub-bass is (1=good)

    # ── Structure ─────────────────────────────────────────────────────────────
    silence_ratio:          float = 0.0   # fraction of near-silent frames
    section_count_estimate: int   = 0     # rough number of distinct sections
    intro_loudness_db:      float = 0.0   # avg RMS of first 8 seconds
    outro_loudness_db:      float = 0.0   # avg RMS of last 8 seconds


# ── Main analysis function ────────────────────────────────────────────────────

def analyse_track(audio: np.ndarray, sr: int, file_path: str = "") -> RichAudioAnalysis:
    """
    Run the full analysis pass on loaded audio.
    audio must be shape (samples, 2) float32, sr = 44100.
    Returns a RichAudioAnalysis dataclass.
    """
    a = RichAudioAnalysis()
    a.file_path       = file_path
    a.sample_rate     = sr
    a.duration_seconds = len(audio) / sr
    a.num_channels    = audio.shape[1] if audio.ndim > 1 else 1

    mono = audio.mean(axis=1).astype(np.float32) if audio.ndim > 1 else audio.astype(np.float32)

    # ── 1. Loudness ────────────────────────────────────────────────────────────
    meter = pyln.Meter(sr)
    try:
        a.integrated_lufs = float(meter.integrated_loudness(audio))
    except Exception:
        a.integrated_lufs = -99.0

    a.true_peak_dbfs = float(20 * np.log10(np.abs(audio).max() + 1e-10))

    rms_val = float(np.sqrt(np.mean(mono ** 2)) + 1e-10)
    a.rms_db = float(20 * np.log10(rms_val))
    peak_val = float(np.abs(mono).max() + 1e-10)
    a.crest_factor_db = float(20 * np.log10(peak_val / rms_val))

    # LRA — difference between loud and quiet short-term LUFS windows
    try:
        hop = int(sr * 3.0)          # 3-second short-term windows
        n_windows = max(2, len(mono) // hop)
        st_lufs = []
        for i in range(n_windows):
            seg = mono[i*hop:(i+1)*hop]
            if len(seg) < sr:
                continue
            seg_2ch = np.stack([seg, seg], axis=1)
            try:
                v = float(meter.integrated_loudness(seg_2ch))
                if np.isfinite(v) and v > -70:
                    st_lufs.append(v)
            except Exception:
                pass
        if len(st_lufs) >= 2:
            st_lufs.sort()
            # LRA = 95th percentile minus 10th percentile of short-term LUFS
            p10 = st_lufs[max(0, int(len(st_lufs) * 0.10))]
            p95 = st_lufs[min(len(st_lufs)-1, int(len(st_lufs) * 0.95))]
            a.loudness_range_lu = round(p95 - p10, 1)
    except Exception:
        a.loudness_range_lu = 0.0

    # ── 2. Dynamics ────────────────────────────────────────────────────────────
    frame_rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=512)[0]
    sorted_rms = np.sort(frame_rms[frame_rms > 1e-6])
    if len(sorted_rms) > 10:
        loud  = np.mean(sorted_rms[int(len(sorted_rms) * 0.9):])
        quiet = np.mean(sorted_rms[:int(len(sorted_rms) * 0.5)])
        a.dynamic_range_db = float(20 * np.log10(loud / (quiet + 1e-10)))
        # Noise floor = quietest 5% of frames
        a.noise_floor_db = float(20 * np.log10(np.mean(sorted_rms[:max(1, int(len(sorted_rms)*0.05))]) + 1e-10))

    # HPSS — harmonic vs percussive energy split
    try:
        y_harm, y_perc = librosa.effects.hpss(mono, margin=3.0)
        total_e = np.mean(mono**2) + 1e-10
        a.harmonic_ratio  = float(np.clip(np.mean(y_harm**2) / total_e, 0, 1))
        a.percussive_ratio = float(np.clip(np.mean(y_perc**2) / total_e, 0, 1))
    except Exception:
        a.harmonic_ratio   = 0.5
        a.percussive_ratio = 0.5

    # Onset (transient) density
    try:
        onsets = librosa.onset.onset_detect(y=mono, sr=sr, units='time')
        a.transient_density_hz = round(float(len(onsets) / max(1, a.duration_seconds)), 2)
    except Exception:
        a.transient_density_hz = 0.0

    # Attack profile — measure RMS 10ms after each onset
    try:
        onset_samples = librosa.onset.onset_detect(y=mono, sr=sr, units='samples')
        attack_window = int(sr * 0.010)  # 10ms
        attack_rms_vals = []
        for o in onset_samples[:50]:  # sample first 50 onsets
            seg = mono[o:o + attack_window]
            if len(seg) == attack_window:
                attack_rms_vals.append(float(np.sqrt(np.mean(seg**2)) + 1e-10))
        if attack_rms_vals:
            a.attack_rms_db = float(20 * np.log10(np.mean(attack_rms_vals)))
    except Exception:
        a.attack_rms_db = a.rms_db

    # ── 3. Spectral — frequency zones ──────────────────────────────────────────
    stft = np.abs(librosa.stft(mono, n_fft=4096, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    power = stft ** 2
    total_power = power.sum() + 1e-10

    for zone, (lo, hi) in FREQ_ZONES.items():
        mask = (freqs >= lo) & (freqs < hi)
        ratio = float(power[mask].sum() / total_power)
        setattr(a, f"{zone}_ratio", round(ratio, 4))

    # 1/3-octave ISO bands
    iso_energies = []
    for fc in ISO_BANDS_HZ:
        lo = fc / (2 ** (1/6))
        hi = fc * (2 ** (1/6))
        mask = (freqs >= lo) & (freqs < hi)
        e = float(power[mask].sum() / total_power) if mask.any() else 0.0
        iso_energies.append(round(e, 6))
    a.iso_band_energies = iso_energies

    # Spectral shape descriptors
    a.spectral_centroid_hz = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))
    a.spectral_rolloff_hz  = float(np.mean(librosa.feature.spectral_rolloff(y=mono, sr=sr, roll_percent=0.85)))
    a.spectral_flatness    = float(np.mean(librosa.feature.spectral_flatness(y=mono)))

    try:
        contrast = librosa.feature.spectral_contrast(y=mono, sr=sr)
        a.spectral_contrast_mean = float(np.mean(contrast))
    except Exception:
        a.spectral_contrast_mean = 0.0

    # MFCCs — 13 coefficients as timbre fingerprint
    try:
        mfccs = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=13)
        a.mfcc_means = [round(float(v), 2) for v in mfccs.mean(axis=1)]
    except Exception:
        a.mfcc_means = [0.0] * 13

    # ── 4. Rhythm ──────────────────────────────────────────────────────────────
    try:
        _, y_percussive = librosa.effects.hpss(mono)
        tempo_arr, beats = librosa.beat.beat_track(y=y_percussive, sr=sr, start_bpm=100.0)
        bpm = float(tempo_arr[0]) if hasattr(tempo_arr, '__len__') else float(tempo_arr)
        # Octave correction
        if 30 < bpm < 70:   bpm *= 2
        elif bpm > 180:     bpm /= 2
        a.detected_bpm = round(bpm, 1)

        # Beat regularity — std dev of inter-beat intervals (lower = more regular)
        if len(beats) > 2:
            beat_times = librosa.frames_to_time(beats, sr=sr)
            ibi = np.diff(beat_times)
            regularity = 1.0 - float(np.clip(np.std(ibi) / (np.mean(ibi) + 1e-6), 0, 1))
            a.beat_regularity = round(regularity, 3)
    except Exception:
        a.detected_bpm    = 0.0
        a.beat_regularity = 0.0

    a.zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(mono)))

    # ── 5. Harmony ─────────────────────────────────────────────────────────────
    try:
        chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
        mean_chroma = chroma.mean(axis=1)
        key_idx = int(np.argmax(mean_chroma))
        key_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        a.key_estimate   = key_names[key_idx % 12]
        a.key_confidence = round(float(mean_chroma[key_idx] / (mean_chroma.sum() + 1e-10)), 3)

        # Mode: compare energy on major vs minor triad intervals from root
        # Major: root + 4 semitones + 7 semitones
        # Minor: root + 3 semitones + 7 semitones
        major_energy = mean_chroma[key_idx] + mean_chroma[(key_idx+4)%12] + mean_chroma[(key_idx+7)%12]
        minor_energy = mean_chroma[key_idx] + mean_chroma[(key_idx+3)%12] + mean_chroma[(key_idx+7)%12]
        a.mode = "major" if major_energy >= minor_energy else "minor"

        # Chord complexity: mean entropy of chroma frames
        chroma_norm = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-10)
        entropy = -np.sum(chroma_norm * np.log2(chroma_norm + 1e-10), axis=0)
        a.chord_complexity = round(float(np.mean(entropy)), 3)
    except Exception:
        a.key_estimate    = "unknown"
        a.mode            = "unknown"
        a.key_confidence  = 0.0
        a.chord_complexity = 0.0

    # ── 6. Stereo ──────────────────────────────────────────────────────────────
    if audio.ndim > 1 and audio.shape[1] == 2:
        L, R = audio[:, 0], audio[:, 1]
        corr = np.corrcoef(L, R)[0, 1]
        a.stereo_correlation = float(np.clip(corr, -1.0, 1.0))

        M = (L + R) * 0.5
        S = (L - R) * 0.5
        mid_rms  = float(np.sqrt(np.mean(M**2)) + 1e-10)
        side_rms = float(np.sqrt(np.mean(S**2)) + 1e-10)
        a.stereo_width = round(float(side_rms / mid_rms), 4)

        # Low-end mono check: correlation of L/R below 200 Hz
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, 200, btype='low', fs=sr, output='sos')
        L_low = sosfiltfilt(sos, L)
        R_low = sosfiltfilt(sos, R)
        low_corr = np.corrcoef(L_low, R_low)[0, 1]
        a.low_end_mono = round(float(np.clip(low_corr, 0, 1)), 3)

    # ── 7. Structure ───────────────────────────────────────────────────────────
    silence_thresh  = 10 ** (-50 / 20.0)  # -50 dBFS
    a.silence_ratio = float(np.mean(np.abs(mono) < silence_thresh))

    # Intro / outro loudness (first and last 8 seconds)
    n8 = min(sr * 8, len(mono) // 3)
    intro = mono[:n8]
    outro = mono[-n8:]
    a.intro_loudness_db = float(20 * np.log10(np.sqrt(np.mean(intro**2)) + 1e-10))
    a.outro_loudness_db = float(20 * np.log10(np.sqrt(np.mean(outro**2)) + 1e-10))

    # Section count via RMS change points
    try:
        rms_env = librosa.feature.rms(y=mono, frame_length=4096, hop_length=2048)[0]
        # Smooth and find major drops/spikes
        from scipy.ndimage import uniform_filter1d
        rms_smooth = uniform_filter1d(rms_env, size=20)
        diff = np.abs(np.diff(rms_smooth))
        threshold = np.percentile(diff, 90)
        a.section_count_estimate = int(np.sum(diff > threshold)) + 1
    except Exception:
        a.section_count_estimate = 1

    return a


def analyse_file(input_path: str | Path) -> RichAudioAnalysis:
    """Convenience: load a file and run full analysis."""
    path = Path(input_path)
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)

    # Ensure stereo
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    elif audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)

    # Resample to 44100
    if sr != 44100:
        import librosa
        L = librosa.resample(audio[:, 0], orig_sr=sr, target_sr=44100)
        R = librosa.resample(audio[:, 1], orig_sr=sr, target_sr=44100)
        audio = np.stack([L, R], axis=1).astype(np.float32)
        sr = 44100

    return analyse_track(audio, sr, str(path))


def to_json(analysis: RichAudioAnalysis, indent: int = 2) -> str:
    """Serialise analysis to compact JSON for the AI advisor."""
    d = asdict(analysis)
    # Round floats for readability
    def _round(v):
        if isinstance(v, float):
            return round(v, 4)
        if isinstance(v, list):
            return [_round(x) for x in v]
        return v
    d = {k: _round(v) for k, v in d.items()}
    return json.dumps(d, indent=indent)