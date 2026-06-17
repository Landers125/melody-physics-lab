"""Акустические признаки, связанные с ознобом (frisson).

Литература связывает озноб с (см. docs/frisson.md):
  - громкость и её рост (крещендо)        — Bannister 2020
  - яркость (спектральный центроид)        — Bannister & Eerola 2018
  - ширина полосы / расширение диапазона  — Guhn 2007
  - шероховатость (roughness proxy)        — Grewe 2007
Возвращается pandas.DataFrame, индекс — время (сек).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None


def _require_librosa():
    if librosa is None:
        raise ImportError("Нужен librosa: pip install librosa")


def spectral_roughness_proxy(S: np.ndarray, freqs: np.ndarray, k_peaks: int = 12) -> np.ndarray:
    """Упрощённый proxy сенсорной шероховатости (Plomp-Levelt / Sethares).

    Для каждого кадра берём top-K спектральных пиков и суммируем вклады пар:
    диссонанс максимален при разности частот ≈ 0.25 критической полосы.
    """
    n_frames = S.shape[1]
    rough = np.zeros(n_frames)
    for t in range(n_frames):
        col = S[:, t]
        if not np.any(col):
            continue
        k = min(k_peaks, col.size)
        idx = np.argpartition(col, -k)[-k:]
        f = freqs[idx]
        a = col[idx]
        order = np.argsort(f)
        f, a = f[order], a[order]
        total = 0.0
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                if f[i] <= 0:
                    continue
                df = f[j] - f[i]
                cbw = 0.24 * (f[i] + df / 2.0) + 25.0  # критическая полоса, Гц
                x = df / cbw
                d = np.exp(-3.5 * x) - np.exp(-5.75 * x)
                if d > 0:
                    total += a[i] * a[j] * d
        rough[t] = total
    if rough.max() > 0:
        rough = rough / rough.max()
    return rough


def extract_audio_features(path: str, sr: int = 22050, hop_length: int = 512,
                           compute_roughness: bool = True) -> pd.DataFrame:
    """Покадровые frisson-признаки из аудиофайла.

    Колонки: rms, loudness_db, centroid, bandwidth, rolloff, flux, roughness,
    d_loudness, d_centroid.
    """
    _require_librosa()
    y, sr = librosa.load(path, sr=sr, mono=True)
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)

    rms = librosa.feature.rms(S=S, hop_length=hop_length)[0]
    loudness_db = librosa.amplitude_to_db(rms + 1e-9)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S + 1e-9), sr=sr)

    n = min(len(rms), len(centroid), len(bandwidth), len(rolloff), len(flux))
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)
    roughness = spectral_roughness_proxy(S[:, :n], freqs) if compute_roughness else np.zeros(n)

    df = pd.DataFrame({
        "time": times,
        "rms": rms[:n],
        "loudness_db": loudness_db[:n],
        "centroid": centroid[:n],
        "bandwidth": bandwidth[:n],
        "rolloff": rolloff[:n],
        "flux": flux[:n],
        "roughness": roughness[:n],
    })
    df["d_loudness"] = df["loudness_db"].diff().fillna(0.0)
    df["d_centroid"] = df["centroid"].diff().fillna(0.0)
    return df.set_index("time")


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
