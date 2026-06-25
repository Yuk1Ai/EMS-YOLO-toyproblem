"""Validate dataset format and the controlled-experiment invariants.

This script is a lightweight quality gate for the generated toy dataset.
It checks both ordinary dataset requirements, such as file existence and
YOLO label format, and the scientific control requirements needed for the
toy problem. In particular, it verifies that the temporal-only condition
differs only by event order, while the control condition contains an
additional visible shape cue.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# EMS-YOLO's event dataloader expects temporal bins with RGB-like channels.
# The generator writes arrays in this layout: (T, H, W, C).
EXPECTED_HEIGHT = 240
EXPECTED_WIDTH = 304
EXPECTED_CHANNELS = 3


def parse_args() -> argparse.Namespace:
    """Parse the dataset root path from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("generated/toy_motion"))
    return parser.parse_args()


def read_label(path: Path) -> tuple[int, np.ndarray]:
    """Read one YOLO-format label file.

    Each label file must contain exactly one object:
    class_id x_center y_center width height
    where the four box coordinates are normalized to [0, 1].
    """
    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 5:
        raise AssertionError(f"Expected five YOLO fields in {path}")
    return int(fields[0]), np.asarray(fields[1:], dtype=np.float64)


def collapsed(events: np.ndarray) -> np.ndarray:
    """Collapse temporal bins into an order-independent event image.

    With neutral=127 and event=255, max over time produces the union of all
    event pixels. This intentionally removes the temporal ordering signal.
    """
    return events.max(axis=0)


def validate_pair(cw_path: Path, condition: str) -> None:
    """Validate one clockwise/counterclockwise paired scene.

    The pair is the core unit of the controlled dataset. Both samples should
    share the same final bounding box and differ only in the intended cue:
    shape cue for the control condition, temporal order for temporal_only.
    """
    ccw_path = cw_path.with_name(cw_path.name.replace("_cw.npy", "_ccw.npy"))

    # Check raw event tensor shape, dtype, and the simplified event encoding.
    cw = np.load(cw_path, allow_pickle=False)
    ccw = np.load(ccw_path, allow_pickle=False)
    assert cw.ndim == 4 and ccw.ndim == 4
    assert cw.shape == ccw.shape
    assert cw.shape[0] >= 2
    assert cw.shape[1:] == (EXPECTED_HEIGHT, EXPECTED_WIDTH, EXPECTED_CHANNELS)
    assert cw.dtype == np.uint8 and ccw.dtype == np.uint8
    assert set(np.unique(cw)).issubset({127, 255})
    assert set(np.unique(ccw)).issubset({127, 255})

    # Check that labels encode direction classes while keeping localization fixed.
    cw_class, cw_box = read_label(cw_path.with_suffix(".txt"))
    ccw_class, ccw_box = read_label(ccw_path.with_suffix(".txt"))
    assert (cw_class, ccw_class) == (0, 1)
    assert np.allclose(cw_box, ccw_box), f"Bounding boxes differ for {cw_path.stem}"

    if condition == "temporal_only":
        # The strict temporal-only invariant: CCW is the reversed CW sequence.
        # Therefore, a model must use order to distinguish the two classes.
        assert np.array_equal(cw, ccw[::-1]), f"Sequences are not reversals: {cw_path}"
        assert np.array_equal(collapsed(cw), collapsed(ccw)), (
            f"Collapsed inputs differ: {cw_path}"
        )
    else:
        # In the control condition, CW and CCW also differ by a static shape cue.
        # The collapsed inputs should therefore not be identical.
        assert not np.array_equal(collapsed(cw), collapsed(ccw)), (
            f"Control shape cue is missing: {cw_path}"
        )


def validate_manifest(condition_root: Path, split: str) -> int:
    """Validate a train.txt or val.txt manifest for one dataset condition."""
    manifest = condition_root / f"{split}.txt"
    paths = [Path(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    assert len(paths) > 0 and len(paths) % 2 == 0
    assert all(p.exists() for p in paths), f"Missing path in {manifest}"

    # The dataset must be balanced between clockwise and counterclockwise classes.
    class_counts = {0: 0, 1: 0}
    for path in paths:
        class_id, _ = read_label(path.with_suffix(".txt"))
        class_counts[class_id] += 1
    assert class_counts[0] == class_counts[1]
    return len(paths)


def main() -> None:
    """Run all validation checks and print a compact split summary."""
    root = parse_args().dataset.resolve()
    summary = {}
    for condition in ("control", "temporal_only"):
        condition_root = root / condition
        summary[condition] = {}

        # Prevent leakage: paired scene IDs must not appear in both train and val.
        train_ids = {p.name.rsplit("_", 1)[0] for p in (condition_root / "train").glob("*_cw.npy")}
        val_ids = {p.name.rsplit("_", 1)[0] for p in (condition_root / "val").glob("*_cw.npy")}
        assert train_ids.isdisjoint(val_ids)

        # Validate every paired scene and cross-check it against the manifest.
        for split in ("train", "val"):
            pair_files = sorted((condition_root / split).glob("*_cw.npy"))
            for cw_path in pair_files:
                validate_pair(cw_path, condition)
            manifest_count = validate_manifest(condition_root, split)
            assert manifest_count == 2 * len(pair_files)
            summary[condition][split] = manifest_count

    print("All dataset checks passed.")
    for condition, counts in summary.items():
        print(f"{condition}: train={counts['train']}, val={counts['val']}")


if __name__ == "__main__":
    main()
