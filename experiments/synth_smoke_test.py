"""Смоук-тест frisson-детектора на СИНТЕТИЧЕСКОМ аудио (без librosa, без реальных файлов).

Зачем: быстрая воспроизводимая проверка логики detect.py / evaluation.py без скачивания
музыки. Генерируем аудио с известными озноб-событиями (крещендо + рост
яркости) плюс «дистракторы» (рост только громкости). Признаки считаются тем же
способом, что в audio_features.py, но через numpy STFT (librosa не нужен).
Матчинг и метрики — настоящие detect.find_peaks / evaluation.

Демонстрирует эффект мягкого joint-гейта (громкость И яркость): он убирает
FP от дистракторов «только громкость». Последний замер: P 0.81->1.0, F1 0.83->0.92.

ВНИМАНИЕ: это синтетика, не настоящая музыка. Для реальной F1 используйте
experiments/run_validation.py на локальном аудио.

Запуск:
    python experiments/synth_smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frisson import detect, evaluation  # noqa: E402

SR = 22050; DUR = 60.0; HOP = 512; FRAME = 2048


def synth_track(events, weak_idx=(), distractors=(), seed=0):
    """Моно-аудио: крещендо+яркость в events; distractors — рост только громкости.

    Детерминированный сид на трек — чтобы обе конфигурации видели одно и то же аудио.
    """
    rng = np.random.default_rng(seed)
    n = int(SR * DUR); t = np.arange(n) / SR
    base = 0.04 * (np.sin(2*np.pi*220*t) + 0.5*np.sin(2*np.pi*330*t))
    base += 0.01 * rng.standard_normal(n)
    env = np.full(n, 0.15); bright = np.zeros(n)
    for i, e in enumerate(events):
        amp = 0.5 if i in weak_idx else 1.0
        c = int(e * SR); ramp = int(1.5 * SR)
        seg = slice(max(0, c - ramp), c)
        k = np.linspace(0, 1, seg.stop - seg.start)
        env[seg] += amp * 0.9 * k; bright[seg] += amp * k
        hold = slice(c, min(n, c + int(0.6 * SR)))
        env[hold] += amp * 0.9; bright[hold] += amp
    for d in distractors:
        c = int(d * SR); ramp = int(1.2 * SR)
        seg = slice(max(0, c - ramp), c)
        k = np.linspace(0, 1, seg.stop - seg.start)
        env[seg] += 0.6 * k
    high = 0.04 * np.sin(2*np.pi*3500*t)
    return (env * base + bright * high).astype(np.float32)


def extract_features_numpy(y, sr=SR, frame=FRAME, hop=HOP):
    """Колонки как у audio_features.extract_audio_features, но без librosa."""
    win = np.hanning(frame); freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    starts = list(range(0, len(y) - frame, hop))
    rms, cen, bw, flux = [], [], [], []; prev_mag = None
    for s in starts:
        seg = y[s:s + frame] * win
        rms.append(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        mag = np.abs(np.fft.rfft(seg)); msum = mag.sum() + 1e-9
        c = float((freqs * mag).sum() / msum); cen.append(c)
        bw.append(float(np.sqrt(((freqs - c) ** 2 * mag).sum() / msum)))
        flux.append(0.0 if prev_mag is None else float(np.clip(mag - prev_mag, 0, None).sum()))
        prev_mag = mag
    rms = np.asarray(rms)
    df = pd.DataFrame({
        "time": np.asarray([s + frame / 2 for s in starts]) / sr,
        "loudness_db": 20 * np.log10(rms), "centroid": np.asarray(cen),
        "bandwidth": np.asarray(bw), "flux": np.asarray(flux),
        "roughness": np.zeros(len(starts)),
    })
    df["d_loudness"] = df["loudness_db"].diff().fillna(0.0)
    df["d_centroid"] = df["centroid"].diff().fillna(0.0)
    return df.set_index("time")


TRACKS = {
    "synth01": dict(events=[6, 14, 23, 31, 38, 46, 52, 57], weak_idx=[4], distractors=[19, 43]),
    "synth02": dict(events=[8, 17, 27, 35, 49], weak_idx=[], distractors=[12]),
    "synth03": dict(events=[5, 12, 20, 29, 40, 50, 56], weak_idx=[2, 5], distractors=[34]),
}
TOL = 1.5


def build_features():
    """Генерируем признаки ОДИН раз (детерминированно), чтобы сравнение было честным."""
    feats = {}
    for si, (tid, c) in enumerate(TRACKS.items()):
        y = synth_track(c["events"], c["weak_idx"], c["distractors"], seed=42 + si)
        feats[tid] = extract_features_numpy(y)
    return feats


def run(feats, require_joint: bool, min_prominence: float, edge_mask_s: float):
    results = []
    for tid, c in TRACKS.items():
        score = detect.frisson_likelihood(feats[tid], require_joint=require_joint)
        peaks = [t for t, _ in detect.find_peaks(
            score, min_prominence=min_prominence, edge_mask_s=edge_mask_s)]
        results.append(evaluation.match_peaks_to_labels(
            peaks, [float(e) for e in c["events"]], TOL))
    return evaluation.aggregate(results)


def main() -> None:
    feats = build_features()
    base = run(feats, require_joint=False, min_prominence=1.0, edge_mask_s=0.0)
    impr = run(feats, require_joint=True, min_prominence=1.0, edge_mask_s=1.0)  # дефолты
    print("=== Синтетический смоук-тест (3 трека, 20 событий, tol=1.5s) ===")
    print(f"baseline (без joint-гейта):  {base.as_dict()}")
    print(f"improved (мягкий joint):   {impr.as_dict()}")
    # Санити-проверки конвейера
    assert impr.recall >= 0.8, "recall упал ниже 0.8"
    assert impr.precision >= base.precision, "precision не вырос"
    assert impr.f1 >= base.f1, "F1 не вырос"
    print("OK: мягкий joint-гейт повышает precision и F1 без обвала recall.")


if __name__ == "__main__":
    main()
