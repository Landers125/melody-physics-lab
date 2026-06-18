"""Standard MIDI File (SMF) reader/writer на чистом Python — без pretty_midi.

fmt 0/1, running status, tempo-map (ticks->секунды), SMPTE division.
read_notes(path) -> list[(start_s, end_s, pitch, velocity)]
read_melody(path) -> list[int] (верхний голос, последовательность высот)
write_notes(path, notes, tpq=480, bpm=120) -> для раунд-трип тестов
"""
from __future__ import annotations

import struct


def _read_vlq(data, i):
    val = 0
    while True:
        b = data[i]
        i += 1
        val = (val << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return val, i


def _write_vlq(value):
    buf = bytearray([value & 0x7F])
    value >>= 7
    while value:
        buf.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(buf)


def read_notes(path):
    with open(path, "rb") as f:
        return parse_notes(f.read())


def parse_notes(data):
    if data[:4] != b"MThd":
        raise ValueError("Not a Standard MIDI File (missing MThd)")
    _hlen, _fmt, _ntrks, division = struct.unpack(">IHHH", data[4:14])
    if division & 0x8000:  # SMPTE
        fps = 256 - ((division >> 8) & 0xFF)  # two's complement of negative frames
        tpf = division & 0xFF
        ticks_per_second = float(fps * tpf)
        ticks_per_qn = None
    else:
        ticks_per_qn = float(division)
        ticks_per_second = None

    pos = 14
    events = []  # (abs_tick, track_idx, kind, payload)
    track_idx = 0
    while pos + 8 <= len(data) and data[pos:pos + 4] == b"MTrk":
        length = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        pos += 8
        end = pos + length
        i = pos
        abs_tick = 0
        status = None
        while i < end:
            dt, i = _read_vlq(data, i)
            abs_tick += dt
            b = data[i]
            if b & 0x80:
                status = b
                i += 1
            if status is None:
                raise ValueError("running status without prior status byte")
            if status == 0xFF:  # meta
                meta_type = data[i]; i += 1
                mlen, i = _read_vlq(data, i)
                payload = data[i:i + mlen]; i += mlen
                if meta_type == 0x51 and mlen == 3:
                    tempo = (payload[0] << 16) | (payload[1] << 8) | payload[2]
                    events.append((abs_tick, track_idx, "tempo", tempo))
            elif status in (0xF0, 0xF7):  # sysex
                slen, i = _read_vlq(data, i)
                i += slen
            else:
                hi = status & 0xF0
                if hi in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                    d1 = data[i]; d2 = data[i + 1]; i += 2
                    if hi == 0x90 and d2 > 0:
                        events.append((abs_tick, track_idx, "on", (status & 0x0F, d1, d2)))
                    elif hi == 0x80 or (hi == 0x90 and d2 == 0):
                        events.append((abs_tick, track_idx, "off", (status & 0x0F, d1)))
                elif hi in (0xC0, 0xD0):
                    i += 1
                else:
                    i += 1
        pos = end
        track_idx += 1

    events.sort(key=lambda e: e[0])
    tempos = sorted((t, val) for (t, _ti, k, val) in events if k == "tempo")

    def t2s(tick):
        if ticks_per_qn is None:
            return tick / ticks_per_second
        sec = 0.0
        last_tick = 0
        tempo = 500000.0  # default 120 bpm
        for (tt, val) in tempos:
            if tt >= tick:
                break
            sec += (tt - last_tick) * tempo / 1e6 / ticks_per_qn
            last_tick = tt
            tempo = val
        sec += (tick - last_tick) * tempo / 1e6 / ticks_per_qn
        return sec

    notes = []
    active = {}
    for (tick, ti, kind, payload) in events:
        if kind == "on":
            ch, pitch, vel = payload
            active.setdefault((ti, ch, pitch), []).append((tick, vel))
        elif kind == "off":
            ch, pitch = payload
            key = (ti, ch, pitch)
            if active.get(key):
                start_tick, vel = active[key].pop(0)
                notes.append((t2s(start_tick), t2s(tick), pitch, vel))
    if events:
        last_tick = events[-1][0]
        for (ti, ch, pitch), lst in active.items():
            for (start_tick, vel) in lst:
                notes.append((t2s(start_tick), t2s(last_tick), pitch, vel))
    notes.sort(key=lambda n: (n[0], n[2]))
    return notes


def notes_to_melody(notes, onset_eps=0.03):
    """Верхний голос: группируем почти-одновременные онсеты, берём самую высокую ноту."""
    if not notes:
        return []
    by_onset = {}
    for (s, e, p, v) in notes:
        key = round(s / onset_eps)
        if key not in by_onset or p > by_onset[key][1]:
            by_onset[key] = (s, p)
    return [p for _k, (s, p) in sorted(by_onset.items())]


def read_melody(path, onset_eps=0.03):
    return notes_to_melody(read_notes(path), onset_eps=onset_eps)


def write_notes(path, notes, tpq=480, bpm=120):
    """Минимальный SMF fmt-0 writer (для тестов раунд-трипа)."""
    us_per_qn = int(round(60_000_000 / bpm))
    track = bytearray()
    track += _write_vlq(0) + bytes([0xFF, 0x51, 0x03,
                                    (us_per_qn >> 16) & 0xFF,
                                    (us_per_qn >> 8) & 0xFF, us_per_qn & 0xFF])
    msgs = []  # (tick, kind, pitch, vel)
    for (s, e, p, v) in notes:
        msgs.append((int(round(s * tpq * bpm / 60)), 0x90, p, v))
        msgs.append((int(round(e * tpq * bpm / 60)), 0x80, p, 0))
    msgs.sort(key=lambda m: (m[0], m[1] == 0x90))  # offs before ons at same tick
    prev = 0
    for (tick, kind, p, v) in msgs:
        track += _write_vlq(tick - prev) + bytes([kind, p & 0x7F, v & 0x7F])
        prev = tick
    track += _write_vlq(0) + bytes([0xFF, 0x2F, 0x00])  # end of track
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq)
    chunk = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    with open(path, "wb") as f:
        f.write(header + chunk)
