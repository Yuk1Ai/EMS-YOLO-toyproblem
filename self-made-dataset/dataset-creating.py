"""Generate a controlled event-like object-detection dataset.

Class 0 moves clockwise and class 1 moves counter-clockwise on the same
closed multi-point trajectory.  In the temporal-only condition both classes
contain exactly the same bins in reverse order.  In the control condition,
the clockwise object is circular and the counter-clockwise object is square.

The generated dataset is designed for a toy controlled experiment around
event order. It uses simplified event-like tensors rather than a full physical
DVS simulation: neutral pixels are encoded as 127 and event pixels as 255.
Each sample is saved as a .npy file with shape (T, H, W, C), and each label is
saved in YOLO format so that the files can be consumed by the EMS-YOLO
g1-resnet dataloader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


# Dataset geometry and event encoding. The EMS-YOLO event dataloader resizes
# frames spatially, while the temporal length is controlled by TIME_STEPS and
# the training-time -T argument.
HEIGHT = 240
WIDTH = 304
TIME_STEPS = 12
NEUTRAL = 127
EVENT = 255
CLASS_NAMES = ["clockwise", "counterclockwise"]

# Difficulty controls. Larger objects and fewer nuisance events make the
# positive-control dataset easier to learn, which is important before testing
# the harder temporal-only hypothesis.
OBJECT_SIZE_MIN = 32
OBJECT_SIZE_MAX = 48  # exclusive upper bound for numpy.integers
ORBIT_RADIUS_MIN = 36
ORBIT_RADIUS_MAX = 64  # exclusive upper bound for numpy.integers
MAX_STATIC_NOISE_EVENTS = 5


def parse_args() -> argparse.Namespace:
    """Parse command-line options controlling output size and randomness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("generated/toy_motion"))
    parser.add_argument("--train-scenes", type=int, default=32)
    parser.add_argument("--val-scenes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def trajectory(rng: np.random.Generator) -> tuple[list[tuple[int, int]], dict]:
    """Sample one closed multi-point orbit and return its scene parameters.

    The object moves around a circle using TIME_STEPS points. The
    counter-clockwise sample later uses the same points in reverse order.
    The label is the bounding box of the whole trajectory, so CW and CCW have
    exactly the same localization target.
    """
    object_size = int(rng.integers(OBJECT_SIZE_MIN, OBJECT_SIZE_MAX))
    object_radius = object_size // 2
    orbit_radius = int(rng.integers(ORBIT_RADIUS_MIN, ORBIT_RADIUS_MAX))

    # Keep the full orbit and object outline inside the image boundary.
    margin = orbit_radius + object_radius + 3
    cx = int(rng.integers(margin, WIDTH - margin))
    cy = int(rng.integers(margin, HEIGHT - margin))

    # Clockwise sequence sampled around a full circle. The negative angle
    # direction follows image coordinates where y grows downward.
    positions = []
    for t in range(TIME_STEPS):
        angle = -2.0 * np.pi * t / TIME_STEPS - np.pi / 2.0
        x = int(round(cx + orbit_radius * np.cos(angle)))
        y = int(round(cy + orbit_radius * np.sin(angle)))
        positions.append((x, y))

    return positions, {
        "center": [cx, cy],
        "orbit_radius": orbit_radius,
        "object_size": object_size,
    }


def render_bin(position: tuple[int, int], object_size: int, shape: str,
               noise_xy: np.ndarray) -> np.ndarray:
    """Render one temporal bin as a simplified event frame.

    The frame contains a neutral background, optional static nuisance events,
    and the current object outline. The output is repeated over three channels
    because the target EMS-YOLO dataloader expects image-like channel layout.
    """
    frame = np.full((HEIGHT, WIDTH), NEUTRAL, dtype=np.uint8)

    # Static nuisance events are shared across paired samples, so they should
    # not reveal the class label.
    if len(noise_xy):
        frame[noise_xy[:, 1], noise_xy[:, 0]] = EVENT

    radius = object_size // 2
    if shape == "circle":
        cv2.circle(frame, position, radius, EVENT, thickness=2, lineType=cv2.LINE_8)
    elif shape == "square":
        x, y = position
        cv2.rectangle(frame, (x - radius, y - radius),
                      (x + radius, y + radius), EVENT, thickness=2,
                      lineType=cv2.LINE_8)
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return np.repeat(frame[..., None], 3, axis=2)


def make_sequence(positions: list[tuple[int, int]], object_size: int,
                  shape: str, noise_xy: np.ndarray) -> np.ndarray:
    """Render all temporal bins for one sample and stack them into a tensor."""
    frames = [render_bin(p, object_size, shape, noise_xy) for p in positions]
    result = np.stack(frames, axis=0)

    # Fail fast if later edits accidentally produce data incompatible with the
    # EMS-YOLO event dataloader.
    assert result.shape == (TIME_STEPS, HEIGHT, WIDTH, 3)
    assert result.dtype == np.uint8
    return result


def yolo_label(class_id: int, positions: list[tuple[int, int]], object_size: int) -> str:
    """Create one YOLO-format detection label for the whole motion pattern.

    The class encodes motion direction: 0 for clockwise and 1 for
    counter-clockwise. The box encloses the whole trajectory rather than one
    frame. This avoids asking the detector to choose one arbitrary temporal
    position from a sequence that contains multiple visible positions.
    """
    radius = object_size // 2 + 2
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    x1 = max(0, min(xs) - radius)
    y1 = max(0, min(ys) - radius)
    x2 = min(WIDTH - 1, max(xs) + radius)
    y2 = min(HEIGHT - 1, max(ys) + radius)
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    box_width = x2 - x1
    box_height = y2 - y1
    return (
        f"{class_id} {x_center / WIDTH:.6f} {y_center / HEIGHT:.6f} "
        f"{box_width / WIDTH:.6f} {box_height / HEIGHT:.6f}\n"
    )


def write_sample(folder: Path, stem: str, events: np.ndarray, label: str) -> Path:
    """Write one sample's .npy tensor and matching .txt YOLO label."""
    npy_path = folder / f"{stem}.npy"
    np.save(npy_path, events, allow_pickle=False)
    (folder / f"{stem}.txt").write_text(label, encoding="utf-8")
    return npy_path.resolve()


def write_manifest(path: Path, samples: list[Path]) -> None:
    """Write a YOLO-style manifest containing absolute .npy sample paths."""
    path.write_text("".join(f"{p.as_posix()}\n" for p in samples), encoding="utf-8")


def write_yaml(root: Path, condition: str) -> None:
    """Write the dataset YAML consumed by EMS-YOLO training and validation."""
    payload = {
        "path": (root / condition).resolve().as_posix(),
        "train": "train",
        "val": "val",
        "nc": 2,
        "names": CLASS_NAMES,
    }
    (root / f"{condition}.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def generate_split(root: Path, split: str, scene_count: int,
                   rng: np.random.Generator, start_id: int) -> tuple[dict, dict]:
    """Generate all paired scenes for one split.

    For each sampled scene, this function writes both dataset conditions:

    - control: clockwise is a circle and counter-clockwise is a square.
    - temporal_only: both directions are circles, and the only discriminative
      signal is the order of temporal bins.

    It returns manifest paths and metadata records so the main function can
    write split manifests and a global metadata file.
    """
    manifests: dict[str, list[Path]] = {"control": [], "temporal_only": []}
    records: dict[str, dict] = {}

    for offset in range(scene_count):
        scene_id = start_id + offset
        scene_key = f"scene_{scene_id:05d}"
        cw_positions, params = trajectory(rng)
        ccw_positions = list(reversed(cw_positions))

        # Static sparse nuisance events are shared by every bin and both classes.
        # They make the frames less empty without becoming a class shortcut.
        noise_count = int(rng.integers(0, MAX_STATIC_NOISE_EVENTS + 1))
        noise_xy = np.column_stack((
            rng.integers(0, WIDTH, size=noise_count),
            rng.integers(0, HEIGHT, size=noise_count),
        )).astype(np.int32)

        for condition in manifests:
            folder = root / condition / split
            cw_shape = "circle"
            ccw_shape = "square" if condition == "control" else "circle"

            # Generate one paired CW/CCW scene under the current condition.
            cw = make_sequence(cw_positions, params["object_size"], cw_shape, noise_xy)
            ccw = make_sequence(ccw_positions, params["object_size"], ccw_shape, noise_xy)

            if condition == "temporal_only":
                # This is the key invariant of the toy problem: after removing
                # order, both classes contain the same event bins.
                assert np.array_equal(cw, ccw[::-1])

            # Each pair has identical final localization but opposite class ID.
            manifests[condition].append(write_sample(
                folder, f"{scene_key}_cw", cw,
                yolo_label(0, cw_positions, params["object_size"]),
            ))
            manifests[condition].append(write_sample(
                folder, f"{scene_key}_ccw", ccw,
                yolo_label(1, ccw_positions, params["object_size"]),
            ))

        records[scene_key] = {
            "split": split,
            **params,
            "positions_clockwise": [list(p) for p in cw_positions],
            "positions_counterclockwise": [list(p) for p in ccw_positions],
            "static_noise_events": noise_count,
        }
    return manifests, records


def main() -> None:
    """Create the complete dataset directory, manifests, YAML files, and metadata."""
    args = parse_args()
    if args.train_scenes < 1 or args.val_scenes < 1:
        raise ValueError("train-scenes and val-scenes must both be positive")

    root = args.output.resolve()
    expected_dirs = [root / c / s for c in ("control", "temporal_only")
                     for s in ("train", "val")]

    # Avoid accidental overwrites because regenerated data could silently change
    # the experiment results.
    if any(d.exists() and any(d.iterdir()) for d in expected_dirs):
        raise FileExistsError(
            f"Refusing to overwrite an existing dataset at {root}. "
            "Choose a new --output path."
        )
    for folder in expected_dirs:
        folder.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    train_manifests, train_records = generate_split(
        root, "train", args.train_scenes, rng, start_id=0
    )
    val_manifests, val_records = generate_split(
        root, "val", args.val_scenes, rng, start_id=args.train_scenes
    )

    # Write one training manifest, one validation manifest, and one dataset YAML
    # for each condition.
    for condition in ("control", "temporal_only"):
        write_manifest(root / condition / "train.txt", train_manifests[condition])
        write_manifest(root / condition / "val.txt", val_manifests[condition])
        write_yaml(root, condition)

    # Metadata records the generation parameters and sampled scene geometry.
    # This helps explain and reproduce the controlled dataset in the report.
    metadata = {
        "description": "Controlled clockwise/counter-clockwise event-like detection data",
        "seed": args.seed,
        "shape": [TIME_STEPS, HEIGHT, WIDTH, 3],
        "encoding": {"neutral": NEUTRAL, "event": EVENT},
        "difficulty": {
            "object_size_range": [OBJECT_SIZE_MIN, OBJECT_SIZE_MAX - 1],
            "orbit_radius_range": [ORBIT_RADIUS_MIN, ORBIT_RADIUS_MAX - 1],
            "max_static_noise_events": MAX_STATIC_NOISE_EVENTS,
        },
        "class_names": CLASS_NAMES,
        "train_scenes": args.train_scenes,
        "val_scenes": args.val_scenes,
        "scenes": {**train_records, **val_records},
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Generated dataset at {root}")
    print(f"Samples: train={2 * args.train_scenes}, val={2 * args.val_scenes}, per condition")


if __name__ == "__main__":
    main()
