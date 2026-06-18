"""Анализ РЕАЛЬНОГО MIDI: H2 (мелодический сюрприз) и H3 (регистр / новый голос).

Без pretty_midi — использует frisson.midi_io (SMF reader на чистом Python).
Печатает пики неожиданности и регистра с таймкодами. Если дана разметка
(CSV с колонкой time_s), считает precision/recall/F1 через frisson.evaluation.

Usage:
  python experiments/run_midi_analysis.py FILE.mid [--order 2] [--min-z 1.5]
      [--win 1.0] [--hop 0.5] [--labels labels.csv] [--tolerance 2.0]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from frisson import midi_io, melodic_surprise as ms, midi_features as mf, evaluation  # noqa: E402


def _zscore(x):
    x = np.asarray(x, dtype=float)
    m = ~np.isnan(x)
    mu, sd = x[m].mean(), x[m].std()
    return (x - mu) / (sd + 1e-9)


def melody_with_times(notes, onset_eps=0.05):
    by = {}
    for s, e, p, v in notes:
        k = round(s / onset_eps)
        if k not in by or p > by[k][1]:
            by[k] = (s, p)
    items = [by[k] for k in sorted(by)]
    return [t for t, _ in items], [p for _, p in items]


def h2_peaks(notes, order=2, alpha=0.1, min_z=1.5):
    times, pitches = melody_with_times(notes)
    an = ms.analyze_melody(pitches, order=order, alpha=alpha)
    if not an:
        return [], pitches
    ic = np.array([a["ic"] for a in an])
    z = _zscore(ic)
    out = []
    for j, a in enumerate(an):
        i = a["index"]
        t = times[i + 1] if i + 1 < len(times) else times[-1]
        is_local_max = (j == 0 or ic[j] >= ic[j - 1]) and (j == len(ic) - 1 or ic[j] >= ic[j + 1])
        if z[j] >= min_z and is_local_max:
            out.append((t, a["interval"], ic[j], z[j]))
    return out, pitches


def h3_peaks(notes, win_s=1.0, hop_s=0.5, min_z=1.0):
    df = mf.windowed_features(notes, win_s=win_s, hop_s=hop_s)
    rt = df["register_top"].to_numpy(dtype=float)
    poly = df["onset_polyphony"].to_numpy(dtype=float)
    tt = df.index.to_numpy()
    comp = np.clip(_zscore(rt), 0, None) + np.clip(_zscore(poly), 0, None)
    out = []
    for k in range(len(comp)):
        if np.isnan(comp[k]):
            continue
        left = comp[k - 1] if k > 0 else -np.inf
        right = comp[k + 1] if k < len(comp) - 1 else -np.inf
        if comp[k] >= min_z and comp[k] >= left and comp[k] >= right:
            out.append((float(tt[k]), float(rt[k]), float(comp[k])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("midi")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--min-z", type=float, default=1.5)
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--tolerance", type=float, default=2.0)
    args = ap.parse_args()

    notes = midi_io.read_notes(args.midi)
    dur = max(n[1] for n in notes)
    print(f"{os.path.basename(args.midi)}: {len(notes)} notes, {dur:.1f}s, "
          f"pitch {min(n[2] for n in notes)}-{max(n[2] for n in notes)}")

    h2, pitches = h2_peaks(notes, order=args.order, alpha=args.alpha, min_z=args.min_z)
    print(f"\nH2  melodic-surprise peaks (z>={args.min_z}, {len(pitches)} melody notes):")
    for t, iv, ic, z in h2:
        print(f"   t={t:6.2f}s  interval={iv:+3d} st  IC={ic:.2f} bits  z={z:.2f}")

    h3 = h3_peaks(notes, win_s=args.win, hop_s=args.hop)
    print(f"\nH3  register/voice peaks:")
    for t, rt, c in h3:
        print(f"   t={t:6.2f}s  register_top={rt:.0f}  composite={c:.2f}")

    if args.labels:
        labels = pd.read_csv(args.labels)["time_s"].astype(float).tolist()
        pred = sorted([t for t, *_ in h2] + [t for t, *_ in h3])
        ev = evaluation.match_peaks_to_labels(pred, labels, tolerance_s=args.tolerance)
        print("\nF1 vs labels:", ev.as_dict())


if __name__ == "__main__":
    main()
