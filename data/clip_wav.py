#!/usr/bin/env python3
"""
Split every .wav in the given folder(s) (recursively) into 4-second clips.
Each output file is named  <stem>_part_000.wav, <stem>_part_001.wav, …

Examples
--------
  python split_folder_wav.py /data/mocap_audio
  python split_folder_wav.py take1.wav take2.wav        # file list
  python split_folder_wav.py /data --dest clips/audio   # save elsewhere
  './wav2folder/Copy of 1_wayne_0_1_1.wav'
"""
from __future__ import annotations
from pathlib import Path
import argparse, sys
from pydub import AudioSegment                 # pydub handles WAV slicing

CLIP_MS = 4000                                # 4 s × 1000 ms

def split_one(src: Path, dest_root: Path) -> None:
    """Write 4-s .wav clips for *src* into *dest_root*."""
    audio = AudioSegment.from_file(src)       # loads at native sample rate
    n_parts = len(audio) // CLIP_MS           # integer division (drop tail)
    if n_parts == 0:
        print(f"[skip] {src.name}: < 4 s")
        return

    stem = src.stem
    for i in range(n_parts):
        start, end = i * CLIP_MS, (i + 1) * CLIP_MS
        part = audio[start:end]               # millisecond slicing :contentReference[oaicite:4]{index=4}
        out = dest_root / f"{stem}_part_{i:03d}.wav"
        part.export(out, format="wav")
        print(f"[ok]  {out.relative_to(dest_root)}  ({CLIP_MS/1000:.0f}s)")

def gather(paths: list[str]) -> list[Path]:
    """Expand folders into *.wav paths and keep explicit files."""
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found.extend(p.rglob("*.wav"))     # recursive pattern :contentReference[oaicite:5]{index=5}
        elif p.is_file():
            found.append(p)
        else:
            print(f"[warn] {p} not found", file=sys.stderr)
    return found

def main(argv=None):
    ap = argparse.ArgumentParser(description="Split WAV files into 4-s clips")
    ap.add_argument("inputs", nargs="+", help="folders and/or .wav files")
    ap.add_argument("--dest", "-d", type=Path,
                    help="output directory (default = alongside source)")
    args = ap.parse_args(argv)

    for src in gather(args.inputs):
        out_dir = args.dest or src.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        split_one(src, out_dir)

if __name__ == "__main__":
    main()
