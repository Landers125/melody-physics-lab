"""Валидация детектора озноба на размеченном корпусе.

Сравнивает предсказанные пики с размеченными моментами озноба и считает
precision / recall / F1 по каждому треку и по корпусу в целом.

Запуск:
    python experiments/run_validation.py data/manifest.csv data/labels.csv \
        [--tolerance 2.0] [--min-prominence 1.0] [--out results.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frisson import audio_features, corpus, detect, evaluation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Валидация frisson-детектора")
    ap.add_argument("manifest")
    ap.add_argument("labels")
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="допуск совпадения по времени, сек")
    ap.add_argument("--min-prominence", type=float, default=1.0,
                    help="порог пика композитной кривой")
    ap.add_argument("--out", default=None, help="CSV с потрековыми метриками")
    args = ap.parse_args()

    tracks = corpus.load_manifest(args.manifest)
    labels = corpus.load_labels(args.labels)

    per_track = []
    results = []
    for tr in tracks:
        gt = labels.get(tr.track_id, [])
        if not gt:
            print(f"[skip] {tr.track_id}: нет разметки")
            continue
        adf = audio_features.extract_audio_features(tr.audio_path)
        score = detect.frisson_likelihood(adf)
        peaks = [t for t, _ in detect.find_peaks(score, min_prominence=args.min_prominence)]
        res = evaluation.match_peaks_to_labels(peaks, gt, tolerance_s=args.tolerance)
        results.append(res)
        row = {"track_id": tr.track_id, "n_pred": len(peaks), "n_label": len(gt), **res.as_dict()}
        per_track.append(row)
        print(f"[ok] {tr.track_id}: P={res.precision:.2f} R={res.recall:.2f} "
              f"F1={res.f1:.2f} (pred={len(peaks)}, label={len(gt)})")

    if results:
        agg = evaluation.aggregate(results)
        print("\n=== ИТОГО по корпусу (микро) ===")
        print(f"  precision={agg.precision:.3f}  recall={agg.recall:.3f}  "
              f"F1={agg.f1:.3f}  median_err={agg.median_error:.2f}s")
        print(f"  TP={agg.tp} FP={agg.fp} FN={agg.fn}")
    else:
        print("Нет треков с разметкой для оценки.")

    if args.out and per_track:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_track[0].keys()))
            w.writeheader()
            w.writerows(per_track)
        print(f"\nПотрековые метрики → {args.out}")


if __name__ == "__main__":
    main()
