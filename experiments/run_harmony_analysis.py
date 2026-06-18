"""Harmonic-tension detector on a real MIDI (frisson H1-harmony branch).

Detects chills driven by harmony (V7, V7b9, bVI color, tritone) that the
melodic-surprise (H2) and register (H3) detectors miss. Optionally scores
precision/recall/F1 against a labels CSV (column time_s).

Usage:
  python experiments/run_harmony_analysis.py FILE.mid [--win 1.0] [--hop 0.5]
      [--min-z 1.0] [--min-gap 2.0] [--labels labels.csv] [--tolerance 2.0]
      [--tonic A] [--mode minor]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from frisson import midi_io, harmony, evaluation  # noqa: E402

PC_NAMES = harmony.PC_NAMES


def _zscore(x):
    x = np.asarray(x, dtype=float)
    m = ~np.isnan(x)
    if m.sum() == 0:
        return np.full_like(x, -np.inf)
    mu, sd = x[m].mean(), x[m].std()
    return (x - mu) / (sd + 1e-9)


def _nms(peaks, value_idx=1, min_gap_s=2.0):
    if not peaks:
        return []
    order = sorted(range(len(peaks)), key=lambda i: peaks[i][value_idx], reverse=True)
    kept = []
    for i in order:
        t = peaks[i][0]
        if all(abs(t - peaks[j][0]) >= min_gap_s for j in kept):
            kept.append(i)
    return [peaks[i] for i in sorted(kept)]


def tension_peaks(df, min_z=1.0, min_gap_s=2.0):
    tt = df.index.to_numpy()
    raw = df["tension"].to_numpy(dtype=float)
    z = _zscore(raw)
    peaks = []
    for k in range(len(z)):
        left = z[k - 1] if k > 0 else -np.inf
        right = z[k + 1] if k < len(z) - 1 else -np.inf
        if z[k] >= min_z and z[k] >= left and z[k] >= right and raw[k] > 0:
            peaks.append((float(tt[k]), float(z[k]), float(raw[k]), df.iloc[k]["bass"]))
    return _nms(peaks, value_idx=1, min_gap_s=min_gap_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("midi")
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--min-z", type=float, default=1.0)
    ap.add_argument("--min-gap", type=float, default=2.0)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--tonic", default=None, help="e.g. A, C#, F")
    ap.add_argument("--mode", default=None, choices=[None, "major", "minor"])
    args = ap.parse_args()

    notes = midi_io.read_notes(args.midi)
    tonic_pc = PC_NAMES.index(args.tonic) if args.tonic else None
    df, tonic_pc, mode = harmony.windowed_tension(
        notes, tonic_pc=tonic_pc, mode=args.mode, win_s=args.win, hop_s=args.hop)
    print(f"{os.path.basename(args.midi)}: {len(notes)} notes, "
          f"key = {PC_NAMES[tonic_pc]} {mode}")

    peaks = tension_peaks(df, min_z=args.min_z, min_gap_s=args.min_gap)
    print(f"\nHARMONIC-tension peaks (z>={args.min_z}):")
    for t, z, raw, bass in peaks:
        row = df.loc[t]
        tags = []
        if row["dominant"]:
            tags.append("V")
        if row["b9_over_bass"]:
            tags.append("b9")
        if row["tritone"]:
            tags.append("tritone")
        if row["leading_tone"]:
            tags.append("LT")
        if row["flatVI"]:
            tags.append("bVI")
        print(f"   t={t:6.2f}s  bass={bass:<2}  tension={raw:.2f}  z={z:.2f}  [{', '.join(tags)}]")

    if args.labels:
        labels = pd.read_csv(args.labels)["time_s"].astype(float).tolist()
        pred = sorted(t for t, *_ in peaks)
        ev = evaluation.match_peaks_to_labels(pred, labels, tolerance_s=args.tolerance)
        print("\nF1 vs labels:", ev.as_dict())


if __name__ == "__main__":
    main()
