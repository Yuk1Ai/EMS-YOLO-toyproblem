"""Create contact sheets and GIF previews for one paired scene per condition.

This script generates quick visual previews of the toy dataset. For each
condition, it selects one validation clockwise/counterclockwise pair and
writes a two-row image: the first row is CW over time, and the second row is
CCW over time. It also writes an animated GIF that shows the same CW/CCW pair
frame by frame. The previews are meant for human inspection, not for training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse dataset and output directories from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("generated/toy_motion"))
    parser.add_argument("--output", type=Path, default=Path("previews"))
    return parser.parse_args()


def annotate(frame: np.ndarray, title: str) -> np.ndarray:
    """Draw a small label on one temporal frame.

    The text is drawn twice, first in black with a thicker stroke and then in
    white with a thinner stroke, so it remains readable on bright event pixels.
    """
    image = frame.copy()
    cv2.putText(image, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def make_pair_gif(cw_events: np.ndarray, ccw_events: np.ndarray, target: Path) -> None:
    """Write an animated GIF comparing CW and CCW motion over time.

    Each GIF frame contains the current CW temporal bin on the top row and the
    corresponding CCW temporal bin on the bottom row. This makes the direction
    cue visible without requiring the reader to inspect the raw .npy arrays.
    """
    frames = []
    for t in range(cw_events.shape[0]):
        cw_frame = annotate(cw_events[t], f"CW t={t}")
        ccw_frame = annotate(ccw_events[t], f"CCW t={t}")
        frames.append(np.vstack([cw_frame, ccw_frame]))

    # Add a short pause at the end before the GIF loops back to t=0.
    frames.extend([frames[-1]] * 2)
    imageio.mimsave(target, frames, duration=0.45, loop=0)


def main() -> None:
    """Build and save preview contact sheets and GIFs for each condition."""
    args = parse_args()
    root = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    for condition in ("control", "temporal_only"):
        # Pick the first validation pair so the preview is deterministic.
        cw_path = next(iter(sorted((root / condition / "val").glob("*_cw.npy"))))
        ccw_path = cw_path.with_name(cw_path.name.replace("_cw.npy", "_ccw.npy"))
        rows = []
        loaded_events = {}
        for class_name, path in (("CW", cw_path), ("CCW", ccw_path)):
            events = np.load(path, allow_pickle=False)
            loaded_events[class_name] = events

            # Horizontally concatenate all temporal bins for one class direction.
            rows.append(np.hstack([
                annotate(events[t], f"{class_name} t={t}") for t in range(events.shape[0])
            ]))

        # Stack CW and CCW rows vertically to compare temporal order directly.
        sheet = np.vstack(rows)
        target = output / f"{condition}_paired_sequence.png"
        cv2.imwrite(str(target), sheet)
        print(f"Wrote {target}")

        gif_target = output / f"{condition}_paired_sequence.gif"
        make_pair_gif(loaded_events["CW"], loaded_events["CCW"], gif_target)
        print(f"Wrote {gif_target}")


if __name__ == "__main__":
    main()
