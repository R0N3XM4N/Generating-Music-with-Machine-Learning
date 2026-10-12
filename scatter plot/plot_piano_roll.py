"""Piano-roll plots of the four generated MIDI files (pitch vs time in beats)."""
from pathlib import Path

import matplotlib.pyplot as plt
from music21 import converter

GEN = Path("artifacts/generated")
MODELS = [
    ("Naive Bayes", "naive_bayes_generated.mid"),
    ("Neural network", "neural_net_generated.mid"),
    ("LSTM", "lstm_generated.mid"),
    ("Encoder-decoder", "encoder_decoder_generated.mid"),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 5))
for ax, (title, fname) in zip(axes.flat, MODELS):
    score = converter.parse(GEN / fname)
    for n in score.flatten().notes:
        pitches = [n.pitch.midi] if n.isNote else [p.midi for p in n.pitches]
        start, dur = float(n.offset), float(n.quarterLength)
        for p in pitches:
            ax.hlines(p, start, start + dur, color="#2b4560", linewidth=2)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("time (beats)")
    ax.set_ylabel("MIDI pitch")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig("generated_piano_rolls.png", dpi=200)
print("saved generated_piano_rolls.png")
