"""Download a bounded, reproducible public classical-piano MIDI subset."""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path


API_URL = "https://api.github.com/repos/cheriell/ClassicalPianoMIDI-dataset/git/trees/main?recursive=1"
RAW_PREFIX = "https://raw.githubusercontent.com/cheriell/ClassicalPianoMIDI-dataset/main/"


def request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "Melodia-ML-educational-project"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Melodia-ML-educational-project"})
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=24, help="number of MIDI files to download")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    print("Fetching dataset inventory...")
    tree = request_json(API_URL).get("tree", [])
    candidates = [item["path"] for item in tree if item.get("type") == "blob" and item["path"].lower().endswith((".mid", ".midi"))]
    if not candidates:
        raise RuntimeError("Dataset inventory did not contain MIDI files. Check network access or the upstream repository.")
    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: min(args.limit, len(candidates))]
    args.output.mkdir(parents=True, exist_ok=True)
    sources = []
    for number, relative_path in enumerate(selected, start=1):
        # Preserve a flat, Windows-friendly local filename while retaining source provenance.
        filename = f"{number:03d}_{Path(relative_path).name}"
        destination = args.output / filename
        url = RAW_PREFIX + relative_path
        if destination.exists() and destination.stat().st_size > 0:
            print(f"Keeping existing {filename}")
        else:
            print(f"Downloading {number}/{len(selected)}: {relative_path}")
            download(url, destination)
        sources.append(f"- `{filename}` <- {url}")
    source_file = args.output.parent / "DATASET_SOURCES.md"
    source_file.write_text(
        "# Dataset provenance\n\n"
        "Downloaded from the public ClassicalPianoMIDI-dataset repository (sourced from Piano-midi.de). "
        "These files are not included in this project; consult the upstream repository and source terms before redistribution.\n\n"
        + "\n".join(sources)
        + "\n",
        encoding="utf-8",
    )
    print(f"Ready: {len(selected)} MIDI files in {args.output}")


if __name__ == "__main__":
    main()
