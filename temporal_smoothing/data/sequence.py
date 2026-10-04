"""Windows of consecutive frames around labeled frames of video sequences."""

import warnings
from typing import Callable, Dict, List, NamedTuple, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["Frame", "SequenceWindowDataset", "BOUNDARY_MODES"]

BOUNDARY_MODES = ("clamp", "skip", "legacy")


class Frame(NamedTuple):
    sequence: str
    index: int
    path: str


class SequenceWindowDataset(Dataset):
    """Yields ``(frames [T, C, H, W], label [H, W])`` for every labeled frame.

    The window holds ``past`` frames before and ``future`` frames after the labeled frame,
    ``T = past + future + 1``, ordered from the oldest to the newest frame. Neighbors are
    taken by position in the sorted frame list of the sequence.

    Args:
        frames: All available video frames.
        labels: Labeled frames; the frame itself must be in ``frames`` (others are skipped).
        load_frame: Loads a frame path into an image tensor.
        load_label: Loads a label path into a label tensor.
        boundary: Windows that extend past the ends of a sequence are ``"clamp"``-ed (the
            first / last frame is repeated), ``"skip"``-ped, or built as in the official
            code (``"legacy"``): from the list of all frames sorted by path, regardless of
            sequence, and skipped only at the ends of that list.
    """

    def __init__(
        self,
        frames: Sequence[Frame],
        labels: Sequence[Frame],
        load_frame: Callable[[str], Tensor],
        load_label: Callable[[str], Tensor],
        past: int = 5,
        future: int = 0,
        boundary: str = "clamp",
    ) -> None:
        if past < 0 or future < 0:
            raise ValueError(f"past and future must be non-negative, got {past}, {future}")
        if boundary not in BOUNDARY_MODES:
            raise ValueError(f"boundary must be one of {BOUNDARY_MODES}, got {boundary}")
        self.load_frame = load_frame
        self.load_label = load_label
        self.past, self.future = past, future

        if boundary == "legacy":
            ordered = {"": sorted(frames, key=lambda f: f.path)}
        else:
            ordered: Dict[str, List[Frame]] = {}
            for frame in sorted(frames, key=lambda f: (f.sequence, f.index)):
                ordered.setdefault(frame.sequence, []).append(frame)
        positions = {
            (frame.sequence, frame.index): (key, pos)
            for key, seq in ordered.items()
            for pos, frame in enumerate(seq)
        }

        self.samples: List[Tuple[List[str], str]] = []
        missing = 0
        for label in sorted(labels, key=lambda f: (f.sequence, f.index)):
            if (label.sequence, label.index) not in positions:
                missing += 1
                continue
            key, pos = positions[(label.sequence, label.index)]
            seq = ordered[key]
            start, end = pos - past, pos + future
            if start < 0 or end >= len(seq):
                if boundary != "clamp":
                    continue
            window = [seq[min(max(i, 0), len(seq) - 1)].path for i in range(start, end + 1)]
            self.samples.append((window, label.path))
        if missing:
            warnings.warn(f"{missing} labeled frames are not among the sequence frames and are skipped")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        window, label_path = self.samples[i]
        loaded: Dict[str, Tensor] = {}  # clamped windows repeat paths
        for path in window:
            if path not in loaded:
                loaded[path] = self.load_frame(path)
        return torch.stack([loaded[path] for path in window]), self.load_label(label_path)
