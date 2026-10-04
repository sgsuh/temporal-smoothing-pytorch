"""Extract consecutive CamVid video frames around labeled frames into ``root/seq``.

The 701-image CamVid release only contains the 1 Hz labeled stills. Temporal smoothing
needs the neighboring 30 Hz frames, which are extracted from the original videos
(http://vis.cs.ucl.ac.uk/Download/G.Brostow/CamVid/)::

    videos/01TP_extract.avi  videos/0006R0.MXF  videos/0005VD.MXF  videos/0016E5.MXF

For every sequence the frame-number offset between the video and the still names is found by
matching the first labeled still against the first frames of the video, and then verified
against every labeled still (train / val / test). Frames are saved as
``seq/{sequence}_{frame}.png`` with the same frame numbering format as the stills.

Usage (inside the container)::

    python scripts/prepare_camvid_seq.py --root /data/CamVid --past 5 --future 2
"""

import argparse
import os
import re
import subprocess
import sys
import zipfile
from typing import Dict, Iterator, List, Tuple

import numpy as np
from PIL import Image

VIDEOS = {
    "0001TP": "01TP_extract.avi",
    "0006R0": "0006R0.MXF",
    "0016E5": "0016E5.MXF",
    "Seq05VD": "0005VD.MXF",
}
SPLITS = ("train", "val", "test")
HEIGHT, WIDTH = 720, 960
_NAME = re.compile(r"^([0-9A-Za-z]+)_([A-Za-z]*)(\d+)\.png$")


def iter_frames(video: str, max_frames: int = 0) -> Iterator[np.ndarray]:
    """Decode a video into ``uint8`` RGB frames ``[720, 960, 3]``."""
    # -vsync 0 passes frames through without duplicating or dropping them (29.97 fps MXF).
    cmd = ["ffmpeg", "-v", "error", "-i", video, "-vsync", "0", "-vf", f"scale={WIDTH}:{HEIGHT}"]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    frame_bytes = HEIGHT * WIDTH * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        while True:
            buffer = proc.stdout.read(frame_bytes)
            if len(buffer) < frame_bytes:
                break
            yield np.frombuffer(buffer, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
        proc.stdout.close()
    if proc.returncode not in (0, None):
        raise RuntimeError(f"ffmpeg failed on {video} (exit code {proc.returncode})")


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def mad(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference."""
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def labeled_stills(root: str) -> Dict[str, Dict[int, Tuple[str, str]]]:
    """``{sequence: {frame: (split, path)}}`` and the frame name format per sequence."""
    stills: Dict[str, Dict[int, Tuple[str, str]]] = {}
    for split in SPLITS:
        directory = os.path.join(root, split)
        for filename in sorted(os.listdir(directory)):
            match = _NAME.match(filename)
            if match:
                sequence, _, number = match.groups()
                stills.setdefault(sequence, {})[int(number)] = (split, os.path.join(directory, filename))
    return stills


def frame_name(sequence: str, number: int, example_path: str) -> str:
    """Format a frame file name like an existing still, e.g. ``0001TP_006690`` or ``Seq05VD_f05100``."""
    _, prefix, digits = _NAME.match(os.path.basename(example_path)).groups()
    return f"{sequence}_{prefix}{number:0{len(digits)}d}.png"


def find_offset(video: str, still: np.ndarray, number: int, search: int, tol: float) -> Tuple[int, float]:
    """Frame-number offset such that ``video index = still number - offset``."""
    diffs = [mad(frame, still) for frame in iter_frames(video, max_frames=search)]
    if not diffs:
        raise RuntimeError(f"no frames decoded from {video}")
    index = int(np.argmin(diffs))
    if diffs[index] > tol:
        raise RuntimeError(
            f"{video}: no frame within the first {search} matches still {number} "
            f"(best index {index}, mean abs diff {diffs[index]:.2f} > {tol})"
        )
    return number - index, diffs[index]


def ensure_video(videos_dir: str, filename: str) -> str:
    path = os.path.join(videos_dir, filename)
    if os.path.exists(path):
        return path
    parts = sorted(p for p in os.listdir(videos_dir) if p.startswith(os.path.splitext(filename)[0] + ".zip."))
    if not parts:
        raise FileNotFoundError(path)
    archive = os.path.join(videos_dir, os.path.splitext(filename)[0] + ".zip")
    if not os.path.exists(archive):
        print(f"joining {parts} -> {archive}", flush=True)
        with open(archive, "wb") as out:
            for part in parts:
                with open(os.path.join(videos_dir, part), "rb") as f:
                    while chunk := f.read(1 << 24):
                        out.write(chunk)
    print(f"extracting {filename} from {archive}", flush=True)
    with zipfile.ZipFile(archive) as zf:
        member = next(m for m in zf.namelist() if os.path.basename(m) == filename)
        with zf.open(member) as src, open(path, "wb") as dst:
            while chunk := src.read(1 << 24):
                dst.write(chunk)
    return path


def process_sequence(args, sequence: str, stills: Dict[int, Tuple[str, str]]) -> bool:
    video = ensure_video(args.videos, VIDEOS[sequence])
    numbers = sorted(stills)
    first = numbers[0]
    offset, first_diff = find_offset(video, load_rgb(stills[first][1]), first, args.search, args.tol)
    print(f"[{sequence}] offset {offset} (still {first} = video frame {first - offset}, diff {first_diff:.2f})", flush=True)

    targets = [n for n in numbers if stills[n][0] in args.splits]
    wanted = {n + d for n in targets for d in range(-args.past, args.future + 1)}
    example = stills[first][1]
    os.makedirs(args.out, exist_ok=True)

    diffs: List[Tuple[int, float]] = []
    # Differences of each still to the previous / next video frame, to check that the matched
    # frame is better than its neighbors (i.e. the offset is not off by one).
    neighbor_diffs: Dict[int, List[float]] = {}
    previous, pending = None, None
    saved = 0
    for index, frame in enumerate(iter_frames(video)):
        number = index + offset
        if pending is not None:
            neighbor_diffs[pending[0]].append(mad(frame, pending[1]))
            pending = None
        if number in stills:
            still = load_rgb(stills[number][1])
            diffs.append((number, mad(frame, still)))
            neighbor_diffs[number] = [mad(previous, still)] if previous is not None else []
            pending = (number, still)
        previous = frame
        if number in wanted:
            path = os.path.join(args.out, frame_name(sequence, number, example))
            if args.overwrite or not os.path.exists(path):
                Image.fromarray(frame).save(path)
            saved += 1

    checked = dict(diffs)
    missing = [n for n in numbers if n not in checked]
    bad = [(n, d) for n, d in diffs if d > args.tol]
    shifted = [(n, d) for n, d in diffs if neighbor_diffs[n] and min(neighbor_diffs[n]) < d]
    neighbors = [min(v) for v in neighbor_diffs.values() if v]
    values = np.array([d for _, d in diffs]) if diffs else np.zeros(1)
    print(
        f"[{sequence}] verified {len(diffs)}/{len(numbers)} stills: mean diff {values.mean():.2f}, "
        f"max {values.max():.2f} (closest neighbor frame: mean {np.mean(neighbors or [0]):.2f}); "
        f"saved {saved}/{len(wanted)} frames",
        flush=True,
    )
    if missing:
        print(f"[{sequence}] stills beyond the video: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if bad:
        print(f"[{sequence}] stills above tolerance {args.tol}: {bad[:10]}{' ...' if len(bad) > 10 else ''}")
    if shifted:
        print(f"[{sequence}] stills closer to a neighbor frame: {shifted[:10]}{' ...' if len(shifted) > 10 else ''}")
    return not bad and not missing and not shifted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.environ.get("CAMVID_ROOT", "/data/CamVid"))
    parser.add_argument("--videos", help="directory of the videos (default: ROOT/videos)")
    parser.add_argument("--out", help="output directory (default: ROOT/seq)")
    parser.add_argument("--past", type=int, default=5, help="frames to keep before each labeled frame")
    parser.add_argument("--future", type=int, default=2, help="frames to keep after each labeled frame")
    parser.add_argument("--splits", nargs="+", default=["test"], choices=SPLITS)
    parser.add_argument("--sequences", nargs="+", default=list(VIDEOS), choices=list(VIDEOS))
    parser.add_argument("--search", type=int, default=2000, help="frames searched for the offset")
    parser.add_argument("--tol", type=float, default=2.0, help="max mean abs difference to a still")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.videos = args.videos or os.path.join(args.root, "videos")
    args.out = args.out or os.path.join(args.root, "seq")

    stills = labeled_stills(args.root)
    ok = True
    for sequence in args.sequences:
        ok &= process_sequence(args, sequence, stills[sequence])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
