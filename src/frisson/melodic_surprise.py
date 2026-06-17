"""Мелодический сюрприз и неопределённость (IDyOM-lite).

Простая n-граммная модель ожидания по мелодическим интервалам со сглаживанием
Лапласа и backoff. Для каждой ноты:
  - information content IC = -log2 p(интервал | контекст)  — "сюрприз"
  - entropy H по распределению следующего интервала        — "неопределённость"
Упрощённый аналог IDyOM (Pearce 2005); baseline для исследования.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class NGramModel:
    order: int = 3
    alpha: float = 0.1  # сглаживание Лапласа
    counts: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    vocab: set = field(default_factory=set)

    def train(self, sequences: list[list[int]]) -> "NGramModel":
        for seq in sequences:
            self.vocab.update(seq)
            for i in range(len(seq)):
                ctx = tuple(seq[max(0, i - self.order):i])
                self.counts[ctx][seq[i]] += 1
        return self

    def prob(self, ctx: tuple[int, ...], symbol: int) -> float:
        ctx = ctx[-self.order:]
        while True:  # backoff по укорачиванию контекста
            table = self.counts.get(ctx)
            if table:
                total = sum(table.values())
                v = max(len(self.vocab), 1)
                return (table.get(symbol, 0) + self.alpha) / (total + self.alpha * v)
            if not ctx:
                return 1.0 / max(len(self.vocab), 1)
            ctx = ctx[1:]

    def distribution(self, ctx: tuple[int, ...]) -> dict[int, float]:
        return {s: self.prob(ctx, s) for s in self.vocab}


def information_content(model: NGramModel, ctx: tuple[int, ...], symbol: int) -> float:
    return -math.log2(max(model.prob(ctx, symbol), 1e-12))


def entropy(model: NGramModel, ctx: tuple[int, ...]) -> float:
    dist = model.distribution(ctx)
    z = sum(dist.values()) or 1.0
    h = 0.0
    for p in dist.values():
        p /= z
        if p > 0:
            h -= p * math.log2(p)
    return h


def pitches_to_intervals(pitches: list[int]) -> list[int]:
    return [pitches[i] - pitches[i - 1] for i in range(1, len(pitches))]


def _score(model: NGramModel, intervals: list[int]):
    out = []
    for i, iv in enumerate(intervals):
        ctx = tuple(intervals[max(0, i - model.order):i])
        out.append({
            "index": i,
            "interval": iv,
            "ic": information_content(model, ctx, iv),
            "entropy": entropy(model, ctx),
        })
    return out


def analyze_melody(pitches: list[int], order: int = 3, alpha: float = 0.1):
    """Обучить модель на самой мелодии и вернуть понотные IC и H."""
    intervals = pitches_to_intervals(pitches)
    model = NGramModel(order=order, alpha=alpha).train([intervals])
    return _score(model, intervals)


def analyze_melody_with_corpus(pitches: list[int], corpus_pitches: list[list[int]],
                               order: int = 3, alpha: float = 0.1):
    """Обучить на внешнем корпусе, оценить целевую мелодию (без утечки)."""
    corpus_intervals = [pitches_to_intervals(p) for p in corpus_pitches]
    model = NGramModel(order=order, alpha=alpha).train(corpus_intervals)
    return _score(model, pitches_to_intervals(pitches))
