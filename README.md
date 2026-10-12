# Generating Music with Machine Learning

This is a symbolic piano-music generation project with four comparable machine-learning approaches:

- `naive_bayes` - smoothed conditional event model
- `neural_net` - fixed-context feed-forward neural network
- `lstm` - recurrent next-event model
- `encoder_decoder` - GRU sequence-to-sequence continuation model

The project runs in three separate, repeatable stages. This is deliberate: preprocessing, training, and generation can be run independently and each stage has a clear input and output.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run each stage

```powershell
# Stage 1: creates artifacts/preprocessed/
python -m src.preprocess

# Stage 2: high-quality model training, typically 45-60 minutes on CPU
python -m src.train_all --preset production --device auto

# Stage 3: writes directly to artifacts/generated/lstm_generated.mid
python -m src.generate --model lstm --events 320 --temperature 0.78 --top-k 12
```

Run Stage 3 again with `naive_bayes`, `neural_net`, or `encoder_decoder` to create their own MIDI files. The recommended `production` preset gives all neural models the same 500 epochs, context length, batch size, optimizer, learning rate, split, vocabulary, and seed. It is designed for a 45-75 minute CPU run on the bundled 24-piece corpus. Each output has a `.generation.json` trace proving which checkpoint sampled its note events.

## Repository map

| Location | Purpose |
|---|---|
| `data/raw/` | 24 downloaded public classical-piano MIDI inputs |
| `src/preprocess.py` | Stage 1: parse, normalize, split, and tokenize |
| `src/train_all.py` | Stage 2: fairly train every model |
| `src/generate.py` | Stage 3: create one polished MIDI output |
| `artifacts/` | Generated preprocessing data, checkpoints, metrics, and output MIDI |
| `PROJECT_GUIDE.md` | Complete commands, inputs/outputs, fairness policy, and troubleshooting |
| `deliverables/` | Write-up PDF and presentation deck for submission |

## Verify

```powershell
python -m pytest -q
```

## Dataset

The included downloader uses a bounded subset of the public [ClassicalPianoMIDI-dataset](https://github.com/cheriell/ClassicalPianoMIDI-dataset). Exact URLs are recorded in `data/DATASET_SOURCES.md`. Review upstream source terms before redistributing data or outputs.
