"""Загрузка корпуса «ознобных» треков и разметки моментов озноба.

Форматы (см. data/README.md):
  manifest.csv: track_id,audio_path,midi_path,artist,title
  labels.csv:   track_id,time_s,source,intensity
      source    — self_report | eda | annotator
      intensity — опц. число (сила отклика), может быть пустым
Стандартная библиотека csv, без внешних зависимостей.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Track:
    track_id: str
    audio_path: str
    midi_path: str | None = None
    artist: str = ""
    title: str = ""


def load_manifest(path: str) -> list[Track]:
    """Читает manifest.csv. Относительные пути разрешаются от папки манифеста."""
    base = Path(path).resolve().parent
    tracks: list[Track] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            audio = row["audio_path"].strip()
            midi = (row.get("midi_path") or "").strip()
            tracks.append(Track(
                track_id=row["track_id"].strip(),
                audio_path=str((base / audio).resolve()) if audio else "",
                midi_path=str((base / midi).resolve()) if midi else None,
                artist=(row.get("artist") or "").strip(),
                title=(row.get("title") or "").strip(),
            ))
    return tracks


def load_labels(path: str) -> dict[str, list[float]]:
    """Читает labels.csv → {track_id: [time_s, ...]} (отсортировано)."""
    out: dict[str, list[float]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["track_id"].strip()].append(float(row["time_s"]))
    return {k: sorted(v) for k, v in out.items()}


def load_labels_detailed(path: str) -> dict[str, list[dict]]:
    """Как load_labels, но сохраняет source и intensity."""
    out: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            intensity = (row.get("intensity") or "").strip()
            out[row["track_id"].strip()].append({
                "time_s": float(row["time_s"]),
                "source": (row.get("source") or "").strip(),
                "intensity": float(intensity) if intensity else None,
            })
    return {k: sorted(v, key=lambda d: d["time_s"]) for k, v in out.items()}
