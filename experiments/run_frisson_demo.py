"""Демо: извлечь frisson-признаки из аудио и найти кандидатные моменты озноба.

Запуск:
    python experiments/run_frisson_demo.py path/to/track.wav [path/to/track.mid]

Вывод: топ кандидатных моментов озноба, окна крещендо и (опц.) пики
мелодического сюрприза по MIDI.
"""
from __future__ import annotations

import sys
from pathlib import Path

# позволяем запуск из корня репозитория
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frisson import audio_features, detect  # noqa: E402


def run(audio_path: str, midi_path: str | None = None, top_k: int = 10) -> None:
    print(f"[1/3] Извлекаю аудио-признаки: {audio_path}")
    adf = audio_features.extract_audio_features(audio_path)

    print("[2/3] Крещендо (рост громкости):")
    for t0, t1, gain in audio_features.detect_crescendos(adf):
        print(f"  {t0:7.2f}s -> {t1:7.2f}s  (+{gain:.1f} dB)")

    print("[3/3] Кандидатные моменты озноба (композитная кривая):")
    score = detect.frisson_likelihood(adf)
    for t, v in detect.find_peaks(score)[:top_k]:
        print(f"  t={t:7.2f}s  score={v:.2f}")

    if midi_path:
        from frisson import midi_features, melodic_surprise
        notes = midi_features.load_notes(midi_path)
        # верхний голос как прокси мелодии
        pitches = [n[2] for n in sorted(notes, key=lambda x: x[0])]
        print("\n[MIDI] Топ мелодического сюрприза (IC):")
        scored = melodic_surprise.analyze_melody(pitches)
        for item in sorted(scored, key=lambda d: d["ic"], reverse=True)[:top_k]:
            print(f"  note#{item['index']:4d}  ic={item['ic']:.2f}  H={item['entropy']:.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
