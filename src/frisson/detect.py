"""Композитный детектор моментов озноба.

Объединяет z-нормированные приросты акустических признаков в единую кривую
"вероятности озноба" и находит локальные пики. Эвристика-baseline
для проверки гипотез H1-H4 (см. docs/frisson.md), а не обученная модель.

Улучшения по итогам синтетической валидации (precision была слабым местом):
  1. edge_mask_s   — отбрасываем пики у самых краёв (артефакты diff/сглаживания);
  2. require_joint — мягкий гейт: фреймы без одновременного роста громкости И
                    яркости понижаются до joint_floor (отсекает «только громкость»
                    — типичный источник FP);
  3. min_prominence / min_gap для отбора пиков.

На синтетическом корпусе (experiments/synth_smoke_test.py): precision 0.81->1.0,
F1 0.83->0.92, FP 4->0, recall сохранён на 0.85.
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


def frisson_likelihood(audio_df: pd.DataFrame, weights: dict | None = None,
                       require_joint: bool = True, joint_eps: float = 0.0,
                       joint_floor: float = 0.5) -> pd.Series:
    """Композитная кривая «вероятности озноба» по аудио-признакам.

    При require_joint=True фреймы, где нет одновременного (по сглаженным
    приростам) роста громкости и яркости, понижаются до joint_floor (не
    обнуляются полностью, чтобы не терять recall). Литература связывает
    озноб именно с сочетанием крещендо и роста яркости.
    """
    weights = weights or DEFAULT_WEIGHTS
    score = pd.Series(0.0, index=audio_df.index)
    for col, w in weights.items():
        if col in audio_df.columns:
            score = score + w * _zscore(audio_df[col]).clip(lower=0)
    if require_joint and {"d_loudness", "d_centroid"} <= set(audio_df.columns):
        zl = _zscore(audio_df["d_loudness"])
        zc = _zscore(audio_df["d_centroid"])
        if len(zl) > 5:  # сглаживаем приросты до гейта: совпадение на масштабе события, а не кадра
            w = max(3, int(len(zl) * 0.01))
            zl = zl.rolling(w, center=True, min_periods=1).mean()
            zc = zc.rolling(w, center=True, min_periods=1).mean()
        rising = ((zl > joint_eps) & (zc > joint_eps)).astype(float)
        # мягкий гейт: вне совместного роста не обнуляем, а понижаем до joint_floor
        gate = joint_floor + (1.0 - joint_floor) * rising
        score = score * gate.to_numpy()
    if len(score) > 5:  # сглаживание
        win = max(3, int(len(score) * 0.01))
        score = score.rolling(win, center=True, min_periods=1).mean()
    return score


def find_peaks(score: pd.Series, min_prominence: float = 1.0,
               min_gap_s: float = 2.0, edge_mask_s: float = 1.0) -> list[tuple[float, float]]:
    """Локальные максимумы. Возвращает [(time, value)] по убыванию value.

    edge_mask_s — игнорируем пики в пределах edge_mask_s от начала/конца.
    """
    if score.empty:
        return []
    times = score.index.to_numpy()
    vals = score.to_numpy()
    t0, t1 = float(times[0]), float(times[-1])
    peaks = []
    for i in range(1, len(vals) - 1):
        t = float(times[i])
        if edge_mask_s > 0 and (t - t0 < edge_mask_s or t1 - t < edge_mask_s):
            continue
        if vals[i] >= vals[i - 1] and vals[i] > vals[i + 1] and vals[i] >= min_prominence:
            peaks.append((t, float(vals[i])))
    peaks.sort(key=lambda x: x[1], reverse=True)
    kept: list[tuple[float, float]] = []
    for t, v in peaks:  # подавление близких пиков
        if all(abs(t - kt) >= min_gap_s for kt, _ in kept):
            kept.append((t, v))
    return kept


def detect_crescendos(df: pd.DataFrame, window_s: float = 3.0,
                      min_db_gain: float = 6.0) -> list[tuple[float, float, float]]:
    """Найти окна устойчивого роста громкости. Возвращает (t_start, t_end, db_gain)."""
    if df.empty:
        return []
    times = df.index.to_numpy()
    loud = df["loudness_db"].to_numpy()
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.05
    win = max(1, int(round(window_s / dt)))
    out: list[tuple[float, float, float]] = []
    i, n = 0, len(loud)
    while i < n - win:
        gain = loud[i + win] - loud[i]
        if gain >= min_db_gain:
            j = i + win
            while j < n - 1 and loud[j + 1] >= loud[j] - 1.0:
                j += 1
            out.append((float(times[i]), float(times[j]), float(loud[j] - loud[i])))
            i = j + 1
        else:
            i += 1
    return out
