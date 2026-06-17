"""Композитный детектор моментов озноба.

Объединяет z-нормированные приросты акустических признаков в единую кривую
"вероятности озноба" и находит локальные пики. Эвристика-baseline
для проверки гипотез H1-H4 (см. docs/frisson.md), а не обученная модель.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - x.mean()) / sd


DEFAULT_WEIGHTS = {
    "d_loudness": 1.0,   # крещендо
    "d_centroid": 0.7,   # рост яркости
    "roughness": 0.5,
    "flux": 0.5,
    "bandwidth": 0.3,    # расширение диапазона
}


def frisson_likelihood(audio_df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Композитная кривая «вероятности озноба» по аудио-признакам."""
    weights = weights or DEFAULT_WEIGHTS
    score = pd.Series(0.0, index=audio_df.index)
    for col, w in weights.items():
        if col in audio_df.columns:
            score = score + w * _zscore(audio_df[col]).clip(lower=0)
    if len(score) > 5:  # сглаживание
        win = max(3, int(len(score) * 0.01))
        score = score.rolling(win, center=True, min_periods=1).mean()
    return score


def find_peaks(score: pd.Series, min_prominence: float = 1.0,
               min_gap_s: float = 2.0) -> list[tuple[float, float]]:
    """Локальные максимумы. Возвращает [(time, value)] по убыванию value."""
    if score.empty:
        return []
    times = score.index.to_numpy()
    vals = score.to_numpy()
    peaks = []
    for i in range(1, len(vals) - 1):
        if vals[i] >= vals[i - 1] and vals[i] > vals[i + 1] and vals[i] >= min_prominence:
            peaks.append((float(times[i]), float(vals[i])))
    peaks.sort(key=lambda x: x[1], reverse=True)
    kept: list[tuple[float, float]] = []
    for t, v in peaks:  # подавление близких пиков
        if all(abs(t - kt) >= min_gap_s for kt, _ in kept):
            kept.append((t, v))
    return kept
