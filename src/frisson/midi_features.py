"""Структурные frisson-признаки из MIDI.

По временным окнам:
  - register_top / register_range — верхний регистр и диапазон (расширение — триггер)
  - note_density    — плотность нот (нарастание фактуры)
  - onset_polyphony — одновременные онсеты (вступление голосов)
Требует pretty_midi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pretty_midi
except ImportError:  # pragma: no cover
    pretty_midi = None


def _require_pm():
    if pretty_midi is None:
        raise ImportError("Нужен pretty_midi: pip install pretty_midi")


def load_notes(midi_path: str) -> list[tuple[float, float, int, int]]:
    """Список нот (start, end, pitch, velocity), отсортированный по start."""
    _require_pm()
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for nt in inst.notes:
            notes.append((nt.start, nt.end, nt.pitch, nt.velocity))
    notes.sort(key=lambda x: x[0])
    return notes


def windowed_features(notes: list[tuple[float, float, int, int]],
                      win_s: float = 1.0, hop_s: float = 0.5) -> pd.DataFrame:
    """Пооконные структурные признаки."""
    if not notes:
        return pd.DataFrame()
    t_end = max(n[1] for n in notes)
    onset_eps = 0.05  # сек: онсеты ближе этого — одновременные
    rows = []
    t = 0.0
    while t < t_end:
        w0, w1 = t, t + win_s
        active = [n for n in notes if n[0] < w1 and n[1] > w0]
        onsets = sorted(n[0] for n in notes if w0 <= n[0] < w1)
        max_simul = 0
        i = 0
        while i < len(onsets):
            j = i
            while j < len(onsets) and onsets[j] - onsets[i] <= onset_eps:
                j += 1
            max_simul = max(max_simul, j - i)
            i = j
        pitches = [n[2] for n in active]
        rows.append({
            "time": t,
            "register_top": max(pitches) if pitches else np.nan,
            "register_low": min(pitches) if pitches else np.nan,
            "register_range": (max(pitches) - min(pitches)) if pitches else 0,
            "note_density": len(onsets) / win_s,
            "onset_polyphony": max_simul,
        })
        t += hop_s
    df = pd.DataFrame(rows).set_index("time")
    df["d_register_top"] = df["register_top"].diff().fillna(0.0)
    df["d_note_density"] = df["note_density"].diff().fillna(0.0)
    return df


def sustained_high_notes(notes, min_dur_s: float = 1.0, top_percentile: float = 90.0):
    """Длинные ноты в верхнем регистре (кандидаты на пик)."""
    if not notes:
        return []
    pitches = np.array([n[2] for n in notes])
    thr = np.percentile(pitches, top_percentile)
    return [(s, e, p) for (s, e, p, v) in notes if (e - s) >= min_dur_s and p >= thr]
