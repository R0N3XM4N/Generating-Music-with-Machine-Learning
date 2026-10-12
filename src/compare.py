"""Generate one consistently configured audition sample for every trained model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import Vocabulary, estimate_key, polish_events, write_midi
from .generate import generate_ids, load_model
from .models import MODEL_NAMES
from .train import CHECKPOINTS, GENERATED, PREPARED, ROOT, choose_device, load_prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.82)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--prepared-dir", type=Path, default=PREPARED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    payload, vocabulary = load_prepared(args.prepared_dir)
    metrics_path = ROOT / "artifacts" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    rows = ["# Model comparison", "", "| Model | Held-out accuracy | Cross-entropy | Audition MIDI |", "|---|---:|---:|---|"]
    for name in MODEL_NAMES:
        checkpoint = CHECKPOINTS / f"{name}.pt"
        if not checkpoint.exists():
            print(f"Skipping {name}; no checkpoint yet.")
            continue
        model, context = load_model(name, device)
        seed_ids = payload["test_sequences"][0][:context]
        seed_events = vocabulary.decode(seed_ids)
        ids = generate_ids(model, seed_ids, args.events, args.temperature, args.top_k, [vocabulary.pad_id, vocabulary.unk_id], context, device)
        path = GENERATED / f"comparison_{name}.mid"
        write_midi(polish_events(vocabulary.decode(ids), estimate_key(seed_events), seed=19), path, f"Melodia comparison {name}")
        result = metrics.get(name, {})
        rows.append(f"| {name} | {result.get('test_accuracy', 'n/a')} | {result.get('test_cross_entropy', 'n/a')} | `{path.relative_to(ROOT)}` |")
        print(f"Wrote {path}")
    report = ROOT / "artifacts" / "comparison.md"
    report.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Comparison report: {report}")


if __name__ == "__main__":
    main()
