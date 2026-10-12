"""Four intentionally different symbolic-music prediction models."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import torch
from torch import Tensor, nn


def sample_logits(logits: Tensor, temperature: float, top_k: int, blocked: Iterable[int]) -> int:
    """Temperature/top-k sampling while excluding non-musical vocabulary symbols."""
    values = logits.detach().float().clone().flatten() / max(temperature, 0.05)
    for token_id in blocked:
        if 0 <= token_id < values.numel():
            values[token_id] = float("-inf")
    if top_k > 0 and top_k < values.numel():
        threshold = torch.topk(values, top_k).values[-1]
        values[values < threshold] = float("-inf")
    probabilities = torch.softmax(values, dim=0)
    if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
        return int(torch.argmax(logits).item())
    return int(torch.multinomial(probabilities, 1).item())


class NaiveBayesMusic:
    """Smoothed P(next_event | recent_events) categorical model.

    It makes the conditional-independence simplification explicit and provides a
    useful transparent baseline against which the neural models can be compared.
    """

    def __init__(self, vocabulary_size: int, context: int = 3, alpha: float = 0.25):
        self.vocabulary_size = vocabulary_size
        self.context = context
        self.alpha = alpha
        self.counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self.global_counts: Counter[int] = Counter()

    def fit(self, sequences: Iterable[list[int]]) -> None:
        for sequence in sequences:
            for index in range(self.context, len(sequence)):
                context = tuple(sequence[index - self.context : index])
                target = int(sequence[index])
                self.counts[context][target] += 1
                self.global_counts[target] += 1

    def distribution(self, history: list[int]) -> Tensor:
        context = tuple(history[-self.context :])
        counts = self.counts.get(context, self.global_counts)
        values = torch.full((self.vocabulary_size,), self.alpha, dtype=torch.float32)
        for index, count in counts.items():
            if 0 <= index < self.vocabulary_size:
                values[index] += count
        return values / values.sum()

    def predict(self, history: list[int]) -> int:
        return int(torch.argmax(self.distribution(history)).item())

    def generate(self, seed: list[int], events: int, temperature: float, top_k: int, blocked: Iterable[int]) -> list[int]:
        output = seed.copy()
        for _ in range(events):
            distribution = self.distribution(output)
            output.append(sample_logits(torch.log(distribution + 1e-9), temperature, top_k, blocked))
        return output

    def state_dict(self) -> dict:
        return {
            "vocabulary_size": self.vocabulary_size,
            "context": self.context,
            "alpha": self.alpha,
            "counts": {"|".join(map(str, key)): dict(value) for key, value in self.counts.items()},
            "global_counts": dict(self.global_counts),
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "NaiveBayesMusic":
        model = cls(payload["vocabulary_size"], payload["context"], payload["alpha"])
        model.counts = defaultdict(Counter)
        for key, value in payload["counts"].items():
            model.counts[tuple(int(item) for item in key.split("|"))] = Counter({int(k): int(v) for k, v in value.items()})
        model.global_counts = Counter({int(k): int(v) for k, v in payload["global_counts"].items()})
        return model


class FeedForwardMusic(nn.Module):
    """Fixed-context next-event classifier (the vanilla neural-network baseline)."""

    def __init__(self, vocabulary_size: int, context: int, embedding_dim: int = 48, hidden_dim: int = 160):
        super().__init__()
        self.config = dict(vocabulary_size=vocabulary_size, context=context, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(context * embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, vocabulary_size),
        )

    def forward(self, source: Tensor) -> Tensor:
        return self.network(self.embedding(source))


class LSTMMusic(nn.Module):
    """Autoregressive LSTM next-event model with long-term recurrence."""

    def __init__(self, vocabulary_size: int, embedding_dim: int = 64, hidden_dim: int = 144, layers: int = 1):
        super().__init__()
        self.config = dict(vocabulary_size=vocabulary_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, layers=layers)
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=layers, batch_first=True, dropout=0.15 if layers > 1 else 0.0)
        self.output = nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, source: Tensor) -> Tensor:
        hidden, _ = self.lstm(self.embedding(source))
        return self.output(hidden[:, -1])


class EncoderDecoderMusic(nn.Module):
    """GRU encoder-decoder trained to continue a multi-event musical phrase."""

    def __init__(self, vocabulary_size: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.config = dict(vocabulary_size=vocabulary_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, source: Tensor, decoder_input: Tensor) -> Tensor:
        _, state = self.encoder(self.embedding(source))
        decoded, _ = self.decoder(self.embedding(decoder_input), state)
        return self.output(decoded)

    @torch.no_grad()
    def generate(self, seed: list[int], events: int, temperature: float, top_k: int, blocked: Iterable[int], device: torch.device) -> list[int]:
        source = torch.tensor([seed], dtype=torch.long, device=device)
        _, state = self.encoder(self.embedding(source))
        current = torch.tensor([[seed[-1]]], dtype=torch.long, device=device)
        output = seed.copy()
        for _ in range(events):
            decoded, state = self.decoder(self.embedding(current), state)
            next_id = sample_logits(self.output(decoded[:, -1])[0], temperature, top_k, blocked)
            output.append(next_id)
            current = torch.tensor([[next_id]], dtype=torch.long, device=device)
        return output


MODEL_NAMES = ("naive_bayes", "neural_net", "lstm", "encoder_decoder")


def build_neural_model(name: str, vocabulary_size: int, context: int) -> nn.Module:
    if name == "neural_net":
        return FeedForwardMusic(vocabulary_size, context)
    if name == "lstm":
        return LSTMMusic(vocabulary_size)
    if name == "encoder_decoder":
        return EncoderDecoderMusic(vocabulary_size)
    raise ValueError(f"Unknown neural model: {name}")


def parameter_count(model: nn.Module) -> int:
    """Trainable parameter count used in the fairness report."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
