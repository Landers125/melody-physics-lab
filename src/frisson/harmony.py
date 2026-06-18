"""Harmonic tension features from MIDI notes (no external deps beyond numpy/pandas).

Motivation (frisson research): melodic-surprise (H2) and register (H3) detectors
miss chills that are driven by *harmony* -- e.g. the lush bVI major7, the
dominant V7, and especially the V7b9 just before a resolution. This module
scores per-window harmonic tension so those moments can be detected directly
from the MIDI pitch content.

Key is estimated with Krumhansl-Kessler profiles; you can also override it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def pc_histogram(notes):
    """Duration-weighted pitch-class histogram (12,)."""
    h = np.zeros(12)
    for s, e, p, v in notes:
        h[p % 12] += max(0.0, e - s)
    return h


def estimate_key(notes):
    """Return (tonic_pc, mode) where mode in {'major','minor'} via KK correlation."""
    h = pc_histogram(notes)
    if h.sum() == 0:
        return 0, "major"
    best = (-2.0, 0, "major")
    for tonic in range(12):
        for prof, mode in ((_KK_MAJOR, "major"), (_KK_MINOR, "minor")):
            rotated = np.roll(prof, tonic)
            r = np.corrcoef(h, rotated)[0, 1]
            if r > best[0]:
                best = (r, tonic, mode)
    return best[1], best[2]


def active_pcs(notes, t, half):
    """(bass_pc, set_of_pcs) sounding within [t-half, t+half]."""
    sounding = [p for (s, e, p, v) in notes if s < t + half and e > t - half]
    if not sounding:
        return None, set()
    bass_pc = min(sounding) % 12
    return bass_pc, set(p % 12 for p in sounding)


def _has_tritone(pcs):
    return any(((a - b) % 12) == 6 for a in pcs for b in pcs)


def tension_components(bass_pc, pcs, tonic_pc, mode):
    """Dict of interpretable harmonic-tension components for one window."""
    if bass_pc is None or not pcs:
        return dict(leading_tone=0, tritone=0, b9_over_bass=0, dominant=0, flatVI=0, dissonance=0.0)
    # leading tone: raised 7th in minor / natural 7th in major == tonic-1
    lt = (tonic_pc - 1) % 12
    leading_tone = 1 if lt in pcs else 0
    tritone = 1 if _has_tritone(pcs) else 0
    b9 = 1 if (bass_pc + 1) % 12 in pcs else 0
    dom_root = (tonic_pc + 7) % 12  # V
    dominant = 1 if bass_pc == dom_root else 0
    # bVI major color (e.g. Fmaj in A minor): root = tonic+8, major triad present
    flatvi_root = (tonic_pc + 8) % 12
    flatVI = 1 if (bass_pc == flatvi_root and (bass_pc + 4) % 12 in pcs and (bass_pc + 7) % 12 in pcs) else 0
    # generic dissonance: count semitone/whole-tone/tritone clashes, normalized
    clashes = 0
    pl = sorted(pcs)
    for i, a in enumerate(pl):
        for b in pl[i + 1:]:
            d = (b - a) % 12
            d = min(d, 12 - d)
            if d in (1, 2, 6):
                clashes += 1
    dissonance = clashes / max(1, len(pl))
    return dict(leading_tone=leading_tone, tritone=tritone, b9_over_bass=b9,
                dominant=dominant, flatVI=flatVI, dissonance=float(dissonance))


DEFAULT_WEIGHTS = dict(leading_tone=1.0, tritone=1.0, b9_over_bass=1.2,
                       dominant=0.5, flatVI=0.8, dissonance=0.4)


def windowed_tension(notes, tonic_pc=None, mode=None, win_s=1.0, hop_s=0.5,
                     weights=None):
    """DataFrame indexed by window time with tension score + components."""
    if tonic_pc is None or mode is None:
        tonic_pc, mode = estimate_key(notes)
    weights = weights or DEFAULT_WEIGHTS
    t_end = max(n[1] for n in notes) if notes else 0.0
    rows = []
    t = 0.0
    half = win_s / 2.0
    while t <= t_end:
        bass, pcs = active_pcs(notes, t, half)
        comp = tension_components(bass, pcs, tonic_pc, mode)
        score = sum(weights[k] * comp[k] for k in weights)
        rows.append(dict(time=t, tension=score,
                         bass=(PC_NAMES[bass] if bass is not None else "-"), **comp))
        t += hop_s
    df = pd.DataFrame(rows).set_index("time")
    return df, tonic_pc, mode
