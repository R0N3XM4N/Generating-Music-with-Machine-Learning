"""Generate a polished piano MIDI continuation from one saved model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .data import Vocabulary, estimate_key, polish_events, write_midi
from .models import EncoderDecoderMusic, FeedForwardMusic, LSTMMusic, NaiveBayesMusic, sample_logits
from .train import CHECKPOINTS, PREPARED, GENERATED, ROOT, choose_device, load_prepared


def load_model(name: str, device: torch.device):
    checkpoint = torch.load(CHECKPOINTS / f"{name}.pt", map_location=device, weights_only=True)
    if "naive_bayes" in checkpoint:
        return NaiveBayesMusic.from_state_dict(checkpoint["naive_bayes"]), checkpoint["context"]
    config = checkpoint["config"]
    model_type = {"neural_net": FeedForwardMusic, "lstm": LSTMMusic, "encoder_decoder": EncoderDecoderMusic}[name]
    model = model_type(**config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["context"]


def generate_ids(model: object, seed: list[int], count: int, temperature: float, top_k: int, blocked: list[int], context: int, device: torch.device) -> list[int]:
    if isinstance(model, NaiveBayesMusic):
        return model.generate(seed, count, temperature, top_k, blocked)
    if isinstance(model, EncoderDecoderMusic):
        return model.generate(seed, count, temperature, top_k, blocked, device)
    output = seed.copy()
    with torch.no_grad():
        for _ in range(count):
            source = torch.tensor([output[-context:]], dtype=torch.long, device=device)
            output.append(sample_logits(model(source)[0], temperature, top_k, blocked))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["naive_bayes", "neural_net", "lstm", "encoder_decoder"], default="lstm")
    parser.add_argument("--events", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.82)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--prepared-dir", type=Path, default=PREPARED, help="output folder from src.preprocess")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.events < 1:
        parser.error("--events must be positive")
    device = choose_device(args.device)
    payload, vocabulary = load_prepared(args.prepared_dir)
    model, context = load_model(args.model, device)
    sequences = payload["test_sequences"] or payload["train_sequences"]
    seed_ids = sequences[args.seed % len(sequences)][:context]
    seed_events = vocabulary.decode(seed_ids)
    ids = generate_ids(
        model, seed_ids, args.events, args.temperature, args.top_k,
        [vocabulary.pad_id, vocabulary.unk_id], context, device,
    )
    output = args.output or GENERATED / f"{args.model}_generated.mid"
    if not output.is_absolute():
        output = ROOT / output
    decoded_events = vocabulary.decode(ids)
    events = polish_events(decoded_events, estimate_key(seed_events), args.seed)
    write_midi(events, output, f"Melodia {args.model} generation")
    checkpoint_digest = hashlib.sha256((CHECKPOINTS / f"{args.model}.pt").read_bytes()).hexdigest()
    trace = {
        "model": args.model,
        "checkpoint_sha256": checkpoint_digest,
        "seed_events_from_prepared_test_corpus": len(seed_ids),
        "model_sampled_events": args.events,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "output_midi": str(output),
        "content_provenance": "All generated pitches and durations are sampled from the saved model. Rendering adds only MIDI-safe range validation and expressive velocity.",
    }
    trace_path = output.with_suffix(".generation.json")
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"Generated {len(events)} events with {args.model}: {output}")
    print(f"Model-generation trace: {trace_path}")


if __name__ == "__main__":
    main()
