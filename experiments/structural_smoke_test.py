"""Воспроизводимый смоук-тест структурных гипотез H2 и H3 на синтетике.

H2 — мелодический сюрприз: пики information content (IC) должны локализовать
     плантированные хроматические скачки (НОВИЗНА, в корпусе нет) и ИГНОРИРОВАТЬ
     «ожидаемые» октавные скачки (частые в корпусе).
H3 — структура MIDI: расширение верхнего регистра + новый голос должны
     локализоваться и НЕ путаться с всплесками плотности без смены регистра.

Нет внешних зависимостей (pretty_midi/librosa): мелодия задаётся списком pitch,
MIDI — списком нот (start, end, pitch, velocity). Используются НАСТОЯЩИЕ
модули frisson.melodic_surprise / midi_features / detect / evaluation.

Запуск:  python experiments/structural_smoke_test.py
Ожидаемый результат (детерминирован): H2 F1=1.0, H3 F1=1.0.

Замечание о порядке модели: на небольшом корпусе лучше работает n-грамма
порядка 1 (сюрприз по предыдущему интервалу): высокие порядки переобучаются
на разрежённых контекстах. Порядок следует повышать вместе с ростом корпуса.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from frisson import melodic_surprise as ms, midi_features as mf, detect, evaluation


# ============================ H2: мелодический сюрприз ============================
def build_h2(rng):
    """Предсказуемая шаговая мелодия с плантированными событиями (через интервалы)."""
    def pitches_from(intervals, start=60):
        p = [start]
        for iv in intervals:
            p.append(int(p[-1] + iv))
        return p

    # Корпус: шаги +-1/+-2 + ЧАСТЫЕ октавные скачки +-7/+-12 (модель их выучивает)
    corpus = []
    for _ in range(10):
        ivs = []
        for _ in range(150):
            ivs.append(int(rng.choice([7, -7, 12, -12])) if rng.random() < 0.12
                       else int(rng.choice([1, -1, 2, -2, 1, -1])))
        corpus.append(pitches_from(ivs))
    # Целевая: строго шаговая +-1/+-2, с плантами
    n = 120
    ivs = [int(rng.choice([1, -1, 2, -2])) for _ in range(n - 1)]
    event_idx = [12, 26, 40, 54, 68, 82, 96, 110]   # результирующая нота после скачка +13
    distractor_idx = [20, 48, 76]                    # «ожидаемый» скачок +7 (выучен)
    for k in event_idx:
        ivs[k - 1] = 13    # хроматический, В КОРПУСЕ НЕ ВСТРЕЧАЕТСЯ
    for k in distractor_idx:
        ivs[k - 1] = 7     # октавный, ЧАСТЫЙ в корпусе
    return pitches_from(ivs), corpus, event_idx, distractor_idx


def run_h2(rng, dt=0.5, order=1, min_prom=1.5):
    target, corpus, event_idx, distractor_idx = build_h2(rng)
    scored = ms.analyze_melody_with_corpus(target, corpus, order=order)
    # индекс интервала i -> нота pitch[i+1], onset время (i+1)*dt
    times = np.array([(s["index"] + 1) * dt for s in scored])
    ic = np.array([s["ic"] for s in scored])
    z_ic = detect._zscore(pd.Series(ic))
    series = pd.Series(z_ic.to_numpy(), index=times)
    peaks = [t for t, _ in detect.find_peaks(series, min_prominence=min_prom,
                                             min_gap_s=2.0, edge_mask_s=1.0)]
    gt = [k * dt for k in event_idx]
    res = evaluation.match_peaks_to_labels(peaks, gt, tolerance_s=1.0)
    zmap = {round(t, 3): v for t, v in zip(times, z_ic.to_numpy())}
    ev_z = float(np.nanmean([zmap.get(round(k * dt, 3), np.nan) for k in event_idx]))
    di_z = float(np.nanmean([zmap.get(round(k * dt, 3), np.nan) for k in distractor_idx]))
    print("--- H2 (мелодический сюрприз) ---")
    print(f"средний z(IC): события={ev_z:.2f}  дистракторы={di_z:.2f}")
    print(f"метрики: {res.as_dict()}")
    return res, ev_z, di_z


# ============================ H3: структура MIDI ============================
def build_h3(rng, dur=60.0):
    notes = []  # (start, end, pitch, velocity)
    t = 0.0
    while t < dur:
        p = 60 + int(rng.integers(0, 5))
        notes.append((t, t + 0.45, p, 70))
        notes.append((t, t + 0.45, p - 5, 65))  # второй голос
        t += 0.5
    events = [7.0, 16.0, 25.0, 34.0, 46.0, 55.0]   # расширение регистра + новый голос
    distractors = [11.0, 40.0]                      # только плотность (без регистра)
    for e in events:
        notes.append((e, e + 1.5, 86 + int(rng.integers(0, 4)), 90))
        notes.append((e, e + 0.45, 79, 80))
        notes.append((e + 0.02, e + 0.45, 74, 78))
    for d in distractors:
        tt = d
        while tt < d + 1.5:
            notes.append((tt, tt + 0.2, 60 + int(rng.integers(0, 5)), 72))
            tt += 0.25
    notes.sort(key=lambda x: x[0])
    return notes, events, distractors


def run_h3(rng, min_prom=1.0):
    notes, events, distractors = build_h3(rng)
    df = mf.windowed_features(notes, win_s=1.0, hop_s=0.5)
    # Композит по H3: рост верхнего регистра + полифония онсетов (НЕ плотность)
    score = (detect._zscore(df["d_register_top"]).clip(lower=0)
             + detect._zscore(df["onset_polyphony"].astype(float)).clip(lower=0))
    if len(score) > 5:
        score = score.rolling(max(3, int(len(score) * 0.05)), center=True, min_periods=1).mean()
    peaks = [t for t, _ in detect.find_peaks(score, min_prominence=min_prom,
                                             min_gap_s=3.0, edge_mask_s=1.0)]
    res = evaluation.match_peaks_to_labels(peaks, events, tolerance_s=1.5)
    print("--- H3 (расширение регистра / новый голос) ---")
    print(f"пики: {[round(p, 1) for p in sorted(peaks)]}")
    print(f"события: {events}  дистракторы: {distractors}")
    print(f"метрики: {res.as_dict()}")
    return res


def main():
    h2, ev_z, di_z = run_h2(np.random.default_rng(7))
    print()
    h3 = run_h3(np.random.default_rng(11))

    # H2: идеальная локализация новизны и чёткое разделение событий и дистракторов
    assert h2.recall == 1.0, h2.as_dict()
    assert h2.f1 == 1.0, h2.as_dict()
    assert ev_z > 2.5 > di_z, (ev_z, di_z)
    # H3: идеальная локализация, дистракторы-плотность отвергнуты
    assert h3.f1 == 1.0, h3.as_dict()
    print("\nOK: H2 и H3 прошли (F1=1.0)")


if __name__ == "__main__":
    main()
