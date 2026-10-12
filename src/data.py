"""MIDI event processing and deterministic conversion back to piano MIDI."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from music21 import chord, converter, instrument, key, meter, note, stream, tempo


REST = "REST"
PAD = "<PAD>"
UNK = "<UNK>"


@dataclass(frozen=True)
class MusicEvent:
    """A simultaneous group of pitches with a duration in quarter lengths."""

    pitches: tuple[int, ...]
    duration: float
    velocity: int = 76

    def token(self) -> str:
        pitch_part = REST if not self.pitches else ".".join(map(str, self.pitches))
        return f"{pitch_part}|{self.duration:.3f}"

    @classmethod
    def from_token(cls, token: str, velocity: int = 76) -> "MusicEvent":
        pitches, duration = token.rsplit("|", 1)
        values = () if pitches == REST else tuple(int(x) for x in pitches.split("."))
        return cls(values, float(duration), velocity)


def quantize_duration(value: float) -> float:
    """Snap to a sixteenth-note grid and constrain pathological MIDI values."""
    return float(np.clip(round(max(value, 0.0625) * 4) / 4, 0.25, 4.0))


def _event_from_element(element: object) -> MusicEvent | None:
    duration = quantize_duration(float(element.duration.quarterLength))
    if isinstance(element, note.Rest):
        return MusicEvent((), duration, 0)
    if isinstance(element, note.Note):
        velocity = int(getattr(element.volume, "velocity", None) or 76)
        return MusicEvent((int(element.pitch.midi),), duration, velocity)
    if isinstance(element, chord.Chord):
        velocity = int(getattr(element.volume, "velocity", None) or 76)
        pitches = tuple(sorted({int(p.midi) for p in element.pitches}))
        return MusicEvent(pitches, duration, velocity)
    return None


def parse_midi(path: Path, max_events: int = 2500) -> list[MusicEvent]:
    """Parse/chordify a MIDI file into a compact event sequence.

    Chordification retains simultaneous harmony while dropping instrument-specific
    tracks, which makes files from mixed sources comparable for a piano model.
    """
    try:
        score = converter.parse(path)
        piano = score.chordify().flatten().notesAndRests
        result = [_event_from_element(element) for element in piano]
        result = [event for event in result if event is not None]
        return result[:max_events]
    except Exception as exc:  # Bad MIDI should not end an entire training run.
        print(f"Skipping unreadable MIDI {path.name}: {exc}")
        return []


def read_corpus(data_dir: Path, min_events: int = 24, max_events_per_file: int = 2500) -> list[list[MusicEvent]]:
    paths = sorted({*data_dir.rglob("*.mid"), *data_dir.rglob("*.midi")})
    pieces = [parse_midi(path, max_events_per_file) for path in paths]
    pieces = [piece for piece in pieces if len(piece) >= min_events]
    if not pieces:
        raise FileNotFoundError(
            f"No usable MIDI files in {data_dir}. Run `python -m src.download_data --limit 24` first."
        )
    return pieces


class Vocabulary:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {index: token for token, index in token_to_id.items()}

    @classmethod
    def build(
        cls, pieces: Iterable[Iterable[MusicEvent]], min_frequency: int = 2, max_tokens: int = 768
    ) -> "Vocabulary":
        counts: dict[str, int] = {}
        for piece in pieces:
            for event in piece:
                token = event.token()
                counts[token] = counts.get(token, 0) + 1
        token_to_id = {PAD: 0, UNK: 1}
        ranked = sorted(
            ((token, count) for token, count in counts.items() if count >= min_frequency),
            key=lambda pair: (-pair[1], pair[0]),
        )[:max_tokens]
        for token, _ in ranked:
            token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    def encode(self, events: Iterable[MusicEvent]) -> list[int]:
        return [self.token_to_id.get(event.token(), self.unk_id) for event in events]

    def decode(self, ids: Iterable[int]) -> list[MusicEvent]:
        events = []
        for index in ids:
            token = self.id_to_token.get(int(index), UNK)
            if token not in {PAD, UNK}:
                events.append(MusicEvent.from_token(token))
        return events

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(path.read_text(encoding="utf-8")))


def split_pieces(pieces: list[list[MusicEvent]], seed: int = 7) -> tuple[list[list[MusicEvent]], list[list[MusicEvent]]]:
    shuffled = pieces.copy()
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * 0.85))
    return shuffled[:cut], shuffled[cut:] or shuffled[-1:]


def windows(sequences: Iterable[list[int]], context: int) -> tuple[np.ndarray, np.ndarray]:
    inputs, targets = [], []
    for sequence in sequences:
        for index in range(context, len(sequence)):
            inputs.append(sequence[index - context : index])
            targets.append(sequence[index])
    if not inputs:
        raise ValueError("Corpus is too small for the chosen sequence length.")
    return np.asarray(inputs, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def estimate_key(events: Iterable[MusicEvent]) -> int:
    """Return a conservative pitch-class tonal centre (C=0) for guardrails."""
    histogram = np.zeros(12, dtype=np.float64)
    for event in events:
        for pitch in event.pitches:
            histogram[pitch % 12] += event.duration
    return int(np.argmax(histogram)) if histogram.sum() else 0


def normalize_to_c(events: list[MusicEvent]) -> tuple[list[MusicEvent], int]:
    """Transpose a piece to C while retaining its intervals and rhythmic shape.

    MIDI collections commonly mix many keys.  A small corpus otherwise makes a
    model spend capacity learning transpositions rather than melodic grammar.
    The original source files remain untouched; only the model representation is
    normalized.  The returned tonic is stored for reproducibility.
    """
    tonic = estimate_key(events)
    normalized = []
    for event in events:
        pitches = tuple(int(np.clip(pitch - tonic, 28, 100)) for pitch in event.pitches)
        normalized.append(MusicEvent(pitches, event.duration, event.velocity))
    return normalized, tonic


MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}


def polish_events(events: list[MusicEvent], tonic: int, seed: int = 7) -> list[MusicEvent]:
    """Add expression without changing the model's sampled musical content.

    Pitch groups and durations remain exactly as predicted by a trained model.
    This function only assigns piano velocities and performs a final MIDI-range
    safety check, making the generated note sequence auditable and model-led.
    """
    rng = random.Random(seed)
    polished: list[MusicEvent] = []
    for event in events:
        if not event.pitches:
            polished.append(MusicEvent((), event.duration, 0))
            continue
        # The parser/model vocabulary keeps pitches in the piano range. Clamping
        # only protects against a malformed external checkpoint; it is not a
        # compositional rule and does not alter valid sampled events.
        pitches_tuple = tuple(int(np.clip(pitch, 21, 108)) for pitch in event.pitches)
        velocity = int(np.clip(67 + 12 * np.sin(len(polished) / 10) + rng.randint(-5, 6), 45, 102))
        polished.append(MusicEvent(pitches_tuple, event.duration, velocity))
    return polished


def write_midi(events: list[MusicEvent], path: Path, title: str = "Melodia ML Generation") -> None:
    """Write a self-contained piano MIDI with tempo, meter and expressive velocities."""
    score = stream.Score(id=title)
    part = stream.Part(id="Piano")
    part.insert(0, instrument.Piano())
    part.insert(0, tempo.MetronomeMark(number=96))
    part.insert(0, meter.TimeSignature("4/4"))
    part.insert(0, key.Key("C"))
    for event in events:
        if not event.pitches:
            element = note.Rest(quarterLength=event.duration)
        elif len(event.pitches) == 1:
            element = note.Note(event.pitches[0], quarterLength=event.duration)
            element.volume.velocity = event.velocity
        else:
            element = chord.Chord(event.pitches, quarterLength=event.duration)
            element.volume.velocity = event.velocity
        part.append(element)
    score.insert(0, part)
    path.parent.mkdir(parents=True, exist_ok=True)
    score.write("midi", fp=str(path))


def save_corpus_summary(pieces: list[list[MusicEvent]], path: Path) -> None:
    summary = {
        "pieces": len(pieces),
        "events": sum(len(piece) for piece in pieces),
        "mean_events_per_piece": round(float(np.mean([len(piece) for piece in pieces])), 1),
        "example": [asdict(event) for event in pieces[0][:4]],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
