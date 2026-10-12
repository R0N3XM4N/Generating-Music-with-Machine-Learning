"""Stage 1: turn raw MIDI files into one reproducible, model-neutral corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import Vocabulary, normalize_to_c, read_corpus, save_corpus_summary, split_pieces


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "artifacts" / "preprocessed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw", help="folder containing .mid or .midi files")
    parser.add_argument("--output-dir", type=Path, default=PREPARED, help="prepared-corpus output folder")
    parser.add_argument("--seed", type=int, default=7, help="fixed split and sampling seed")
    parser.add_argument("--min-frequency", type=int, default=2, help="minimum train occurrence for a vocabulary event")
    parser.add_argument("--max-vocabulary", type=int, default=768, help="maximum event-token vocabulary size")
    parser.add_argument("--max-events-per-file", type=int, default=2500, help="cap very long MIDI files fairly")
    args = parser.parse_args()
    if args.max_vocabulary < 32:
        parser.error("--max-vocabulary must be at least 32")

    pieces = read_corpus(args.data_dir, min_events=24, max_events_per_file=args.max_events_per_file)
    normalized, tonics = [], []
    for piece in pieces:
        item, tonic = normalize_to_c(piece)
        normalized.append(item)
        tonics.append(tonic)
    train_pieces, test_pieces = split_pieces(normalized, args.seed)
    vocabulary = Vocabulary.build(train_pieces, args.min_frequency, args.max_vocabulary)
    train_ids = [vocabulary.encode(piece) for piece in train_pieces]
    test_ids = [vocabulary.encode(piece) for piece in test_pieces]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary.save(args.output_dir / "vocabulary.json")
    payload = {
        "format": "melodia-preprocessed-v1",
        "seed": args.seed,
        "train_sequences": train_ids,
        "test_sequences": test_ids,
        "normalization": "each piece transposed to C using duration-weighted pitch-class tonic estimate",
        "source_piece_count": len(pieces),
        "train_piece_count": len(train_pieces),
        "test_piece_count": len(test_pieces),
        "original_tonics": tonics,
        "vocabulary_size": len(vocabulary.token_to_id),
    }
    serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    (args.output_dir / "corpus.json").write_text(serialized, encoding="utf-8")
    payload["sha256"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    (args.output_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_corpus_summary(normalized, args.output_dir / "corpus_summary.json")
    print("Preprocessing complete")
    print(f"  source MIDI pieces: {len(pieces)}")
    print(f"  split: {len(train_pieces)} train / {len(test_pieces)} test")
    print(f"  normalized vocabulary: {len(vocabulary.token_to_id)} tokens")
    print(f"  prepared corpus: {args.output_dir / 'corpus.json'}")


if __name__ == "__main__":
    main()
