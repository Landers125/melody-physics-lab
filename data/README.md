# Корпус «ознобных» треков и разметка

Эта папка описывает, как собрать данные для проверки гипотез H1-H4 (см. `docs/frisson.md`).

## Аудиофайлы не коммитятся

Музыка защищена авторским правом — `*.wav`, `*.mp3`, `*.mid` игнорируются (см. `.gitignore`).
Храните аудио локально и ссылайтесь на них через манифест. В репозиторий идёт
только метаданные и разметка.

## Источники разметки

- **ChiM dataset** и обзор de Fleurian & Pearce (2021), *Chills in Music: A Systematic Review* — список треков с упоминаниями озноба.
- **Self-report**: испытуемые нажимают кнопку в момент озноба (source=`self_report`).
- **EDA / GSR**: пики кожно-гальванической реакции (source=`eda`).
- **Экспертная**: ручная разметка аннотатором (source=`annotator`).

## Форматы

### `manifest.csv`

```
track_id,audio_path,midi_path,artist,title
t001,audio/t001.wav,midi/t001.mid,Artist Name,Track Title
```

Пути разрешаются относительно папки манифеста. `midi_path` необязателен.

### `labels.csv`

```
track_id,time_s,source,intensity
t001,42.5,self_report,3
t001,88.0,eda,
```

- `time_s` — момент озноба в секундах от начала трека.
- `intensity` — опциональная сила отклика (может быть пустой).

## Запуск валидации

```bash
python experiments/run_validation.py data/manifest.csv data/labels.csv \
    --tolerance 2.0 --min-prominence 1.0 --out results.csv
```

Скрипт считает precision / recall / F1 и медианную ошибку совпадения по каждому
треку и по корпусу. Допуск ±2 с соответствует типичной задержке между музыкальным
событием и субъективным отчётом.
