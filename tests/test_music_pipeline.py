from pathlib import Path

from src.data import MusicEvent, Vocabulary, estimate_key, parse_midi, polish_events, quantize_duration, windows, write_midi
from src.models import NaiveBayesMusic


def test_duration_quantization():
    assert quantize_duration(0.37) == 0.25
    assert quantize_duration(1.61) == 1.5


def test_vocabulary_windows_and_naive_bayes():
    piece = [MusicEvent((60,), 1.0), MusicEvent((62,), 1.0), MusicEvent((64,), 1.0)] * 4
    vocabulary = Vocabulary.build([piece], min_frequency=1)
    sequence = vocabulary.encode(piece)
    source, target = windows([sequence], 3)
    assert source.shape[0] == len(piece) - 3
    model = NaiveBayesMusic(len(vocabulary.token_to_id), context=3)
    model.fit([sequence])
    assert model.predict(sequence[:3]) == target[0]


def test_midi_roundtrip_and_polishing(tmp_path: Path):
    source = [MusicEvent((60, 64, 67), 1.0), MusicEvent((62,), 0.5), MusicEvent((), 0.5)]
    output = tmp_path / "example.mid"
    write_midi(polish_events(source, estimate_key(source)), output)
    parsed = parse_midi(output)
    assert output.exists() and output.stat().st_size > 0
    assert len(parsed) >= 2


def test_polishing_preserves_model_notes_and_durations():
    source = [MusicEvent((60, 64, 67), 1.0), MusicEvent((62,), 0.5)]
    polished = polish_events(source, estimate_key(source))
    assert [event.pitches for event in polished] == [event.pitches for event in source]
    assert [event.duration for event in polished] == [event.duration for event in source]
