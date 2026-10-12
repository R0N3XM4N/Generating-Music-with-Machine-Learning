"""Train one or all Melodia ML models and create audition-ready MIDI samples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .data import Vocabulary, windows
from .models import MODEL_NAMES, EncoderDecoderMusic, NaiveBayesMusic, build_neural_model, parameter_count


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "artifacts" / "checkpoints"
PREPARED = ROOT / "artifacts" / "preprocessed"
GENERATED = ROOT / "artifacts" / "generated"


def choose_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation.")
    return torch.device("cuda" if choice in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    correct = valid_total = loss_total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for source, target in loader:
            logits = model(source.to(device))
            target = target.to(device)
            loss_sum += float(loss_function(logits, target).item()) * target.numel()
            # Unknown events were intentionally hidden from the train vocabulary;
            # reporting an easy "UNK" hit would inflate held-out accuracy.
            valid = target != 1
            correct += int(((logits.argmax(dim=-1) == target) & valid).sum().item())
            valid_total += int(valid.sum().item())
            loss_total += target.numel()
    return correct / max(valid_total, 1), loss_sum / max(loss_total, 1)


def train_next_event(name: str, train_sequences: list[list[int]], test_sequences: list[list[int]], vocabulary: Vocabulary, args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, dict]:
    source, target = windows(train_sequences, args.sequence_length)
    test_source, test_target = windows(test_sequences, args.sequence_length)
    loader = DataLoader(TensorDataset(Tensor(source).long(), Tensor(target).long()), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(Tensor(test_source).long(), Tensor(test_target).long()), batch_size=args.batch_size)
    model = build_neural_model(name, len(vocabulary.token_to_id), args.sequence_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    best_state, best_loss = None, float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_count = 0
        for batch_source, batch_target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_source.to(device))
            batch_target = batch_target.to(device)
            loss = loss_function(logits, batch_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * batch_target.numel()
            total_count += batch_target.numel()
        test_accuracy, test_loss = accuracy(model, test_loader, device)
        print(f"{name:15s} epoch {epoch:02d}/{args.epochs}: train CE={total_loss/max(total_count,1):.3f}, test CE={test_loss:.3f}, accuracy={test_accuracy:.3f}")
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {"test_accuracy": round(test_accuracy, 4), "test_cross_entropy": round(best_loss, 4)}


def seq2seq_windows(sequences: list[list[int]], context: int, horizon: int = 8) -> tuple[np.ndarray, np.ndarray]:
    sources, targets = [], []
    for sequence in sequences:
        for index in range(context, len(sequence) - horizon + 1):
            sources.append(sequence[index - context : index])
            targets.append(sequence[index : index + horizon])
    if not sources:
        raise ValueError("Corpus is too small for encoder-decoder context/horizon.")
    return np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def train_encoder_decoder(train_sequences: list[list[int]], test_sequences: list[list[int]], vocabulary: Vocabulary, args: argparse.Namespace, device: torch.device) -> tuple[EncoderDecoderMusic, dict]:
    source, target = seq2seq_windows(train_sequences, args.sequence_length)
    test_source, test_target = seq2seq_windows(test_sequences, args.sequence_length)
    loader = DataLoader(TensorDataset(Tensor(source).long(), Tensor(target).long()), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(Tensor(test_source).long(), Tensor(test_target).long()), batch_size=args.batch_size)
    model = build_neural_model("encoder_decoder", len(vocabulary.token_to_id), args.sequence_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    best_state, best_loss, final_accuracy = None, float("inf"), 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_count = 0
        for batch_source, batch_target in loader:
            batch_source, batch_target = batch_source.to(device), batch_target.to(device)
            decoder_input = torch.cat([batch_source[:, -1:], batch_target[:, :-1]], dim=1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_source, decoder_input)
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), batch_target.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * batch_target.numel()
            total_count += batch_target.numel()
        model.eval()
        correct = valid_count = token_count = 0
        test_loss_sum = 0.0
        with torch.no_grad():
            for batch_source, batch_target in test_loader:
                batch_source, batch_target = batch_source.to(device), batch_target.to(device)
                decoder_input = torch.cat([batch_source[:, -1:], batch_target[:, :-1]], dim=1)
                logits = model(batch_source, decoder_input)
                loss = loss_function(logits.reshape(-1, logits.shape[-1]), batch_target.reshape(-1))
                test_loss_sum += float(loss.item()) * batch_target.numel()
                valid = batch_target != vocabulary.unk_id
                correct += int(((logits.argmax(-1) == batch_target) & valid).sum().item())
                valid_count += int(valid.sum().item())
                token_count += batch_target.numel()
        test_loss = test_loss_sum / max(token_count, 1)
        final_accuracy = correct / max(valid_count, 1)
        print(f"encoder_decoder epoch {epoch:02d}/{args.epochs}: train CE={total_loss/max(total_count,1):.3f}, test CE={test_loss:.3f}, accuracy={final_accuracy:.3f}")
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {"test_accuracy": round(final_accuracy, 4), "test_cross_entropy": round(best_loss, 4)}


def train_naive_bayes(train_sequences: list[list[int]], test_sequences: list[list[int]], vocabulary: Vocabulary) -> tuple[NaiveBayesMusic, dict]:
    model = NaiveBayesMusic(len(vocabulary.token_to_id))
    model.fit(train_sequences)
    correct = total = 0
    for sequence in test_sequences:
        for index in range(model.context, len(sequence)):
            if sequence[index] != vocabulary.unk_id:
                correct += model.predict(sequence[index - model.context : index]) == sequence[index]
                total += 1
    return model, {"test_accuracy": round(correct / max(total, 1), 4), "test_cross_entropy": None}


def save_checkpoint(name: str, model: object, context: int) -> None:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS / f"{name}.pt"
    if isinstance(model, NaiveBayesMusic):
        torch.save({"name": name, "context": context, "naive_bayes": model.state_dict()}, path)
    else:
        torch.save({"name": name, "context": context, "config": model.config, "state_dict": model.state_dict()}, path)


PROFILES = {
    "quality": {"epochs": 18, "sequence_length": 32, "batch_size": 64, "learning_rate": 1e-3},
    "quick": {"epochs": 2, "sequence_length": 16, "batch_size": 64, "learning_rate": 2e-3},
    # Measured on the bundled CPU: the three neural models take roughly
    # 45-75 minutes in total. Runtime varies by CPU/GPU and data size.
    "production": {"epochs": 500, "sequence_length": 40, "batch_size": 64, "learning_rate": 5e-4},
}


def _apply_profile(args: argparse.Namespace) -> None:
    profile = PROFILES[args.preset]
    for key, value in profile.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def load_prepared(prepared_dir: Path) -> tuple[dict, Vocabulary]:
    corpus_path = prepared_dir / "corpus.json"
    vocab_path = prepared_dir / "vocabulary.json"
    if not corpus_path.exists() or not vocab_path.exists():
        raise FileNotFoundError(
            f"Prepared corpus not found in {prepared_dir}. Run `python -m src.preprocess` before training."
        )
    return json.loads(corpus_path.read_text(encoding="utf-8")), Vocabulary.load(vocab_path)


def main(forced_model: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODEL_NAMES], default=forced_model or "all")
    parser.add_argument("--prepared-dir", type=Path, default=PREPARED)
    parser.add_argument("--preset", choices=sorted(PROFILES), default="quality", help="quality targets a sub-hour CPU run; quick is a smoke test")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if forced_model:
        args.model = forced_model
    _apply_profile(args)
    if args.sequence_length < 8:
        parser.error("--sequence-length must be at least 8")
    seed_everything(args.seed)
    device = choose_device(args.device)
    print(f"Using {device}.")
    payload, vocabulary = load_prepared(args.prepared_dir)
    train_sequences = payload["train_sequences"]
    test_sequences = payload["test_sequences"]
    vocab_path = CHECKPOINTS / "vocabulary.json"
    vocabulary.save(vocab_path)
    print(f"Loaded prepared corpus: {payload['train_piece_count']} train / {payload['test_piece_count']} test pieces, vocabulary {len(vocabulary.token_to_id)} tokens.")
    print(f"Fair schedule: preset={args.preset}, epochs={args.epochs}, context={args.sequence_length}, batch={args.batch_size}, learning-rate={args.learning_rate}.")
    names = MODEL_NAMES if args.model == "all" else (args.model,)
    metrics = {}
    for name in names:
        if name == "naive_bayes":
            model, metric = train_naive_bayes(train_sequences, test_sequences, vocabulary)
        elif name == "encoder_decoder":
            model, metric = train_encoder_decoder(train_sequences, test_sequences, vocabulary, args, device)
        else:
            model, metric = train_next_event(name, train_sequences, test_sequences, vocabulary, args, device)
        save_checkpoint(name, model, args.sequence_length)
        metrics[name] = {
            **metric,
            "preset": args.preset,
            "epochs": args.epochs if name != "naive_bayes" else "not applicable (closed-form counts)",
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size if name != "naive_bayes" else "not applicable",
            "parameter_count": parameter_count(model) if isinstance(model, nn.Module) else 0,
            "prepared_corpus_sha256": (args.prepared_dir / "manifest.json").exists() and json.loads((args.prepared_dir / "manifest.json").read_text()).get("sha256"),
        }
        print(f"Saved {name} checkpoint.")
    metrics_path = ROOT / "artifacts" / "metrics.json"
    existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    existing.update(metrics)
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
