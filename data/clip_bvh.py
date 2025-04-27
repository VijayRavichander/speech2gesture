#!/usr/bin/env python3
"""
Split every .bvh in one or more folders (recursively) into 4-second clips.
Each output file has exactly one MOTION header with the right frame count.
"""
from __future__ import annotations
from pathlib import Path
import argparse, sys

FRAME_TIME       = 0.008333             # seconds per frame (120 fps)
FRAMES_PER_CLIP  = int(4 / FRAME_TIME)  # 4 s → 480 frames

def split_one(src: Path, dest_root: Path) -> None:
    header, rows = [], []
    with src.open() as f:
        for line in f:
            if line.startswith("MOTION"):
                break                    # stop BEFORE copying MOTION
            header.append(line)

        _ = f.readline()                # throw away original “Frames: …”
        frame_time_line = f.readline()  # keep “Frame Time: …” text
        rows = f.readlines()            # motion data

    clips = len(rows) // FRAMES_PER_CLIP
    if clips == 0:
        print(f"[skip] {src.name}: < 4 s")
        return

    stem = src.stem
    for i in range(clips):
        start, stop = i*FRAMES_PER_CLIP, (i+1)*FRAMES_PER_CLIP
        out = dest_root / f"{stem}_part_{i:03d}.bvh"
        with out.open("w") as w:
            w.writelines(header)
            w.write("MOTION\n")
            w.write(f"Frames: {FRAMES_PER_CLIP}\n")
            w.write(frame_time_line)
            w.writelines(rows[start:stop])
        print(f"[ok] {out.relative_to(dest_root)} ({stop-start} frames)")

def gather(targets: list[str]) -> list[Path]:
    found = []
    for t in map(Path, targets):
        if t.is_dir():
            found.extend(t.rglob("*.bvh"))          # recursive search :contentReference[oaicite:1]{index=1}
        elif t.is_file():
            found.append(t)
    return found

def main(argv=None):
    ap = argparse.ArgumentParser(description="Split BVH files into 4-s clips")
    ap.add_argument("inputs", nargs="+", help="folders and/or .bvh files")
    ap.add_argument("--dest", "-d", type=Path,
                    help="output directory (default = alongside source)")
    args = ap.parse_args(argv)

    for src in gather(args.inputs):
        out_dir = args.dest or src.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        split_one(src, out_dir)

if __name__ == "__main__":
    main()
