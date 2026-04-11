"""
KetMaster CLI — ketmaster.py
=============================
Usage:
    python ketmaster.py song.mp3
    python ketmaster.py song.mp3 --platform spotify --api-key sk-xxx
    python ketmaster.py song.mp3 --platform beatport --output mastered/
    python ketmaster.py song.mp3 --platform all
    python ketmaster.py song.mp3 --no-ai          (rule-based, no API key needed)

Set your DeepSeek API key:
    export DEEPSEEK_API_KEY=sk-your-key-here
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import asdict

from mastering_engine import master_track, PLATFORM_TARGETS, MasteringReport


# ── Colour helpers ────────────────────────────────────────────────────────────
def _c(code, text):   return f"\033[{code}m{text}\033[0m"
def green(t):         return _c("32", t)
def yellow(t):        return _c("33", t)
def red(t):           return _c("31", t)
def cyan(t):          return _c("36", t)
def bold(t):          return _c("1",  t)
def dim(t):           return _c("2",  t)
def magenta(t):       return _c("35", t)
def blue(t):          return _c("34", t)

def _bar(v, lo, hi, w=20, fill="█", empty="░"):
    r = max(0.0, min(1.0, (v - lo) / max(1e-6, hi - lo)))
    f = int(r * w)
    return fill * f + empty * (w - f)


# ── Report printer ────────────────────────────────────────────────────────────
def print_report(report: MasteringReport) -> None:
    b  = report.before
    a  = report.after
    r  = report.recipe

    print()
    print(bold("━" * 64))
    print(bold(f"  🎛  KETMASTER  ─  {report.platform.upper()} MASTER"))
    print(bold("━" * 64))

    print(f"\n  {bold('INPUT')}    {dim(report.input_path)}")
    print(f"  {bold('OUTPUT')}   {dim(report.output_path)}")
    print(f"  {bold('TARGET')}   {report.target_lufs:.0f} LUFS  /  {report.target_true_peak:.1f} dBTP")
    print(f"  {bold('TIME')}     {report.elapsed_s:.1f}s")

    # ── AI Context ───────────────────────────────────────────────────────────
    if r:
        ai_tag = dim("(rule-based)") if r.used_fallback else cyan(f"(DeepSeek {r.model_used})")
        print(f"\n  {bold('AI ANALYSIS')}  {ai_tag}")
        print(f"  {'─'*58}")
        print(f"  Genre       {cyan(r.detected_genre)}")
        print(f"  Mood        {r.detected_mood}")
        print(f"  Energy      {r.energy_level}   |   Vocals: {r.vocal_presence}   |   {r.acoustic_or_electronic}")
        if r.producer_note:
            print(f"\n  {dim('Producer note:')}")
            # Word-wrap at 56 chars
            words = r.producer_note.split()
            line, lines = [], []
            for w in words:
                if len(' '.join(line + [w])) > 56:
                    lines.append(' '.join(line))
                    line = [w]
                else:
                    line.append(w)
            if line:
                lines.append(' '.join(line))
            for ln in lines:
                print(f"  {dim(ln)}")

    # ── Before / After ───────────────────────────────────────────────────────
    print(f"\n  {bold('METRICS'):<28}  {'BEFORE':>10}  {'AFTER':>10}  {'DELTA':>8}")
    print(f"  {'─'*58}")

    def row(label, bv, av, fmt=".1f", invert=False):
        try:
            delta = av - bv
            d_str = f"{delta:+.1f}"
            col   = green if (delta > 0) != invert else red
            if abs(delta) < 0.15:
                col = dim
            print(f"  {label:<26}  {bv:>10{fmt}}  {av:>10{fmt}}  {col(d_str):>16}")
        except Exception:
            print(f"  {label:<26}  {'N/A':>10}  {'N/A':>10}  {'—':>8}")

    row("Integrated LUFS",      b.integrated_lufs,   a.integrated_lufs)
    row("True peak (dBFS)",     b.true_peak_dbfs,    a.true_peak_dbfs)
    row("Dynamic range (dB)",   b.dynamic_range_db,  a.dynamic_range_db)
    row("Loudness range (LU)",  b.loudness_range_lu, a.loudness_range_lu)
    row("RMS (dB)",             b.rms_db,            a.rms_db)
    row("Crest factor (dB)",    b.crest_factor_db,   a.crest_factor_db, invert=True)
    row("Stereo correlation",   b.stereo_correlation, a.stereo_correlation, fmt=".3f")
    row("Stereo width",         b.stereo_width,      a.stereo_width, fmt=".3f")
    row("Spectral centroid Hz", b.spectral_centroid_hz, a.spectral_centroid_hz, fmt=".0f")
    row("Transient density/s",  b.transient_density_hz, a.transient_density_hz, fmt=".1f")

    # ── Track identity ───────────────────────────────────────────────────────
    print(f"\n  {bold('TRACK IDENTITY')}")
    print(f"  {'─'*58}")
    print(f"  BPM          {b.detected_bpm:.1f}   |   Beat regularity: {b.beat_regularity:.2f}")
    print(f"  Key          {b.key_estimate} {b.mode}   (confidence: {b.key_confidence:.2f})")
    print(f"  Duration     {b.duration_seconds:.1f}s  ({b.duration_seconds/60:.2f} min)")
    print(f"  Noise floor  {b.noise_floor_db:.1f} dBFS")
    print(f"  Harmonic     {b.harmonic_ratio:.2f}   |   Percussive: {b.percussive_ratio:.2f}")
    print(f"  Low-end mono {b.low_end_mono:.3f}  {'✓' if b.low_end_mono > 0.8 else '⚠'}")

    # ── Frequency balance ────────────────────────────────────────────────────
    print(f"\n  {bold('FREQUENCY BALANCE  (before → after)')}")
    zones = [
        ("Sub bass <80Hz",  "sub_bass_ratio",  0, 0.3),
        ("Bass 80-200Hz",   "bass_ratio",      0, 0.4),
        ("Low-mid 200-500", "low_mid_ratio",   0, 0.4),
        ("Mid 500-2k",      "mid_ratio",       0, 0.5),
        ("Upper-mid 2-5k",  "upper_mid_ratio", 0, 0.3),
        ("Air >5kHz",       "air_ratio",       0, 0.2),
    ]
    for label, attr, lo, hi in zones:
        bv = getattr(b, attr, 0) * 100
        av = getattr(a, attr, 0) * 100
        print(f"  {label:<18}  {_bar(bv/100, lo, hi)}  {bv:>4.1f}%  →  {av:.1f}%")

    # ── Chain decisions ──────────────────────────────────────────────────────
    print(f"\n  {bold('MASTERING CHAIN DECISIONS')}")
    print(f"  {'─'*58}")

    if report.corrective_eq_applied:
        for m in report.corrective_eq_applied:
            db_str = f" {m.get('db',0):+.1f}dB" if m.get('type') != 'highpass' else " cut"
            print(f"  {yellow('Corrective EQ')}  {m.get('type','peak'):>10}  {m['hz']:>6.0f}Hz{db_str}  — {dim(m.get('reason',''))}")
    else:
        print(f"  {green('Corrective EQ')}   no problem frequencies found")

    if r:
        c = r.comp
        print(f"  {cyan('Glue Comp')}      {c.ratio:.1f}:1  attack:{c.attack_ms:.0f}ms  release:{c.release_ms:.0f}ms  GR:{report.glue_comp_gr_db:+.2f}dB")
        print(f"  {cyan('M/S Width')}      side: {report.ms_side_db:+.1f}dB  mono below {r.ms.low_mono_cutoff_hz:.0f}Hz")

    if report.tonal_eq_applied:
        for m in report.tonal_eq_applied:
            print(f"  {magenta('Tonal EQ')}       {m.get('type','peak'):>10}  {m['hz']:>6.0f}Hz  {m['db']:+.1f}dB  — {dim(m.get('reason',''))}")

    print(f"  {magenta('Saturation')}     drive={report.saturation_drive:.2f}")
    print(f"  {bold('Normaliser')}     gain applied: {report.limiter_gain_db:+.2f}dB")

    if r and r.reasoning:
        print(f"\n  {bold('AI REASONING')}")
        print(f"  {'─'*58}")
        words = r.reasoning.split()
        line, chars = [], 0
        for w in words:
            if chars + len(w) > 56:
                print(f"  {dim(' '.join(line))}")
                line, chars = [w], len(w)
            else:
                line.append(w)
                chars += len(w) + 1
        if line:
            print(f"  {dim(' '.join(line))}")

    # ── Warnings ─────────────────────────────────────────────────────────────
    if report.warnings:
        print(f"\n  {bold('WARNINGS')}")
        for w in report.warnings:
            print(f"  {red(w)}")

    # ── QA ───────────────────────────────────────────────────────────────────
    print()
    if report.passed_qa:
        print(bold(green("  ✓  QA PASSED  —  Ready for distribution")))
    else:
        print(bold(red("  ✗  QA FAILED  —  Review warnings above")))

    print(bold("━" * 64))
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog="ketmaster",
        description="AI-driven professional mastering — powered by DeepSeek",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input",
        help="Input audio file (WAV, MP3, FLAC, AIFF, OGG)")
    parser.add_argument("--platform", "-p", default="spotify",
        choices=list(PLATFORM_TARGETS.keys()) + ["all"],
        help="Target platform")
    parser.add_argument("--output", "-o", default=None,
        help="Output path (default: <input>_mastered.wav)")
    parser.add_argument("--api-key", default=None,
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--no-ai", action="store_true",
        help="Skip AI, use rule-based fallback (no API key needed)")
    parser.add_argument("--lufs",      type=float, default=None)
    parser.add_argument("--true-peak", type=float, default=None)
    parser.add_argument("--json",      action="store_true",
        help="Save full JSON report alongside output")
    parser.add_argument("--quiet",     action="store_true")
    parser.add_argument("--project-dir", "-d", default=None,
        help="Parent directory containing 'originals/' and 'mastered/' subfolders. "
             "If not set, defaults to input file's parent directory.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(red(f"Error: file not found: {input_path}"))
        sys.exit(1)

    platforms = list(PLATFORM_TARGETS.keys()) if args.platform == "all" else [args.platform]
    api_key   = args.api_key or os.environ.get("DEEPSEEK_API_KEY")

    # ------------------------------------------------------------------
    # Determine project root and originals/mastered directories (FIXED)
    # ------------------------------------------------------------------
    if args.project_dir:
        project_root = Path(args.project_dir).resolve()
    else:
        # If input is already inside an 'originals' folder, use its parent as project root
        if input_path.parent.name == "originals":
            project_root = input_path.parent.parent.resolve()
        else:
            project_root = input_path.parent.resolve()

    originals_dir = project_root / "originals"
    mastered_dir  = project_root / "mastered"

    originals_dir.mkdir(parents=True, exist_ok=True)
    mastered_dir.mkdir(parents=True, exist_ok=True)

    # Move input into originals/ only if not already there
    target_input = originals_dir / input_path.name
    if input_path != target_input:
        print(f"  Moving input to {target_input}")
        shutil.move(str(input_path), str(target_input))
        input_path = target_input

    # ------------------------------------------------------------------
    # Process each requested platform
    # ------------------------------------------------------------------
    for platform in platforms:
        stem = input_path.stem
        suffix = f"_{platform}" if len(platforms) > 1 else "_mastered"
        out_path = mastered_dir / f"{stem}{suffix}.wav"

        if not args.quiet:
            print(f"\n  {dim('Processing')} {cyan(input_path.name)}  →  {bold(platform.upper())}")

        report = master_track(
            input_path       = input_path,
            output_path      = out_path,
            platform         = platform,
            api_key          = api_key,
            custom_lufs      = args.lufs,
            custom_true_peak = args.true_peak,
            force_fallback   = args.no_ai,
        )

        if args.quiet:
            status = "PASS" if report.passed_qa else "FAIL"
            print(f"{out_path}  [{status}  {report.after.integrated_lufs:.1f}LUFS  {report.after.true_peak_dbfs:.1f}dBTP]")
        else:
            print_report(report)

        if args.json:
            json_path = out_path.with_suffix(".json")
            data = {
                "input":    report.input_path,
                "output":   report.output_path,
                "platform": report.platform,
                "passed_qa": report.passed_qa,
                "before":   report.before.__dict__,
                "after":    report.after.__dict__,
                "recipe":   report.recipe.__dict__ if report.recipe else {},
                "warnings": report.warnings,
            }
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            if not args.quiet:
                print(f"  {dim('JSON saved to')} {json_path}\n")

if __name__ == "__main__":
    main()