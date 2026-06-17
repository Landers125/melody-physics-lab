"""Оценка детектора озноба против разметки.

Сопоставляем предсказанные пики (из detect.find_peaks) с размеченными моментами
озноба (self-report / EDA-пики) в пределах допуска по времени.
Метрики: precision / recall / F1, а также медианная ошибка совпадения.

Проверяет гипотезы H1-H4 (см. docs/frisson.md): насколько хорошо признаки
предсказывают фактические моменты озноба.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalResult:
    tp: int
    fp: int
    fn: int
    matched_errors: list[float]  # |t_pred - t_label| для совпавших пар

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def median_error(self) -> float:
        if not self.matched_errors:
            return float("nan")
        s = sorted(self.matched_errors)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0

    def as_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "median_error_s": round(self.median_error, 3) if self.matched_errors else None,
        }


def match_peaks_to_labels(predicted_times: list[float], label_times: list[float],
                          tolerance_s: float = 2.0) -> EvalResult:
    """Жадное одно-к-одному сопоставление ближайших пар в пределах допуска.

    Каждая метка и каждый пик используются не более одного раза.
    """
    preds = sorted(predicted_times)
    labels = sorted(label_times)
    # все возможные пары в пределах допуска, отсортированные по ошибке
    pairs = []
    for pi, pt in enumerate(preds):
        for li, lt in enumerate(labels):
            err = abs(pt - lt)
            if err <= tolerance_s:
                pairs.append((err, pi, li))
    pairs.sort()
    used_pred: set[int] = set()
    used_label: set[int] = set()
    errors: list[float] = []
    for err, pi, li in pairs:
        if pi in used_pred or li in used_label:
            continue
        used_pred.add(pi)
        used_label.add(li)
        errors.append(err)
    tp = len(errors)
    fp = len(preds) - tp
    fn = len(labels) - tp
    return EvalResult(tp=tp, fp=fp, fn=fn, matched_errors=errors)


def aggregate(results: list[EvalResult]) -> EvalResult:
    """Микро-агрегация по корпусу (суммируем TP/FP/FN)."""
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    errors: list[float] = []
    for r in results:
        errors.extend(r.matched_errors)
    return EvalResult(tp=tp, fp=fp, fn=fn, matched_errors=errors)
