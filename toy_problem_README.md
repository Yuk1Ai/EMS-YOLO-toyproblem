# Toy problem: Deep Directly-Trained Spiking Neural Networks for Object Detection

## Model hypthesis for the self created dataset

The aim of this control experiment is to classify clockwise(CW) motion and counter clockwise(CCW) motion. The only controled difference of the dataset is the motion orientation. In the self maded dataset, tempol order is the only reliable difference. Whether the EMS-YOLO model could detect the correct motion orientation is the control experiment tryin to figure out.

### The goal of this dataset:

EMS-YOLO can use the temporal order of event bins to detect the objects motion.

### Positive result:

For those events with same space content but reversed time order, the EMS-YOLO can successfully detect the clockwise(CW) and counter clockwise(CCW) motions. But when temporal order is collapsed, for example use T=1 instead of T=4, the detecting performance of EMS-YOLO model will decrease.

### Dataset and model

Dataset:

* The dataset should ensure that the space contents of CW and CCW.
* The start point, end point, bounding box, number of events are the same.
* The only reliable difference is the temporal order of the 12 event bins.

Model:
Can EMS-YOLO use this unique difference to perform motion-direction detection?

### Core control

* Catagory 0: Clockwise motion
* Catagory 1:Counter clockwise motion
* The object will back to the same end point every 12 time step.
* The CW and CCW dataset use 12 same event frame
* Same position of the final bounding box

Therefore, the dataset should have strong control: the catagory can only be classified by the temporal order, could not leak from the final position, background, speed or event amounts.

### Two dataset conditions

**Control data**

* Different shape of the objects. CW is circle, CCW is square.
* Can be classified when the temporal information is lost
* used to prove the reliability of the model

**Temporal_only data**

* Same shape of objects for both CW and CCW,circle
* Only difference is the temporal order
* If the temporal order is ==collapsed==, the model will have nearly random classification performance theoretically

Apart from the different temporal order, the number of samples, trajectory, location, background, amount of events, target size, and label distribution are all the same.

## Dataset creation

Each samle is a ``.npy``tensor, the shape is :

```Python

(12, 240, 304, 3)
```

Which means each sample contains 12 temporal bins, $240 \times 304$ space resolution and 3 channel input to support EMS-YOLO dataloader image-like input format.

The object will move along a circle trajectory make up by 12 points. YOLO bounding box surrounds the entire motion trajectory so that the detector can detect the object location.

### Dataset generation

The file `self-made-dataset/dataset-creating.py` generates the controlled dataset. For each sampled scene, it creates a clockwise sample and a counter-clockwise sample with the same trajectory, same bounding box, same noise events, and opposite temporal order. In the `control` condition, CW uses a circle and CCW uses a square. In the `temporal_only` condition, both classes use circles, so the class can only be identified from the order of the temporal bins.

Key generation code:

```Python
cw_positions, params = trajectory(rng)
ccw_positions = list(reversed(cw_positions))

noise_count = int(rng.integers(0, MAX_STATIC_NOISE_EVENTS + 1))
noise_xy = np.column_stack((
    rng.integers(0, WIDTH, size=noise_count),
    rng.integers(0, HEIGHT, size=noise_count),
)).astype(np.int32)

for condition in manifests:
    folder = root / condition / split
    cw_shape = "circle"
    ccw_shape = "square" if condition == "control" else "circle"

    cw = make_sequence(cw_positions, params["object_size"], cw_shape, noise_xy)
    ccw = make_sequence(ccw_positions, params["object_size"], ccw_shape, noise_xy)

    if condition == "temporal_only":
        assert np.array_equal(cw, ccw[::-1])

    manifests[condition].append(write_sample(
        folder, f"{scene_key}_cw", cw,
        yolo_label(0, cw_positions, params["object_size"]),
    ))
    manifests[condition].append(write_sample(
        folder, f"{scene_key}_ccw", ccw,
        yolo_label(1, ccw_positions, params["object_size"]),
    ))
```

Install dataset generation requirements:

```PowerShell
pip install -r self-made-dataset\requirements.txt
```

Generate dataset:

```PowerShell
python self-made-dataset\dataset-creating.py `
  --output self-made-dataset\generated\toy_motion_12pt_clean `
  --train-scenes 200 `
  --val-scenes 50 `
  --seed 42
```

Generate preview figure and GIF:

```PowerShell
python self-made-dataset\visualize_samples.py `
  --dataset self-made-dataset\generated\toy_motion_12pt_clean `
  --output self-made-dataset\previews\toy_motion_12pt_clean
```

## Control experiment

### Model use

Using the smaller EMS-ResNet10, computational cost is low:

* Model configuration: [`resnet10.yaml`](g1-resnet/models/resnet10.yaml)
* Training entry point: [`train_g1.py`](g1-resnet/train_g1.py)
* Validation entry point: [`val.py`](g1-resnet/val.py)
* Data loader: [`datasets_g1T.py`](g1-resnet/utils/datasets_g1T.py)

### Experiment matrix

| Experiment                      | Dataset condition               | Model / input                     | Check target and expected result                                                                                    |
| ------------------------------- | ------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `control_T4_12pt_clean`       | `control`, with shape cue     | EMS-YOLO, ordered`T=4`          | High performance. This verifies that the dataset, labels, bounding boxes, and training pipeline are learnable.      |
| `temporal_only_T4_12pt_clean` | `temporal_only`, no shape cue | EMS-YOLO, ordered`T=4`          | High performance. This verifies the model can classify the CW motion and CCW motion of the dataset.                 |
| `temporal_only_T1_12pt_clean` | `temporal_only`, no shape cue | EMS-YOLO, order-collapsed,`T=1` | Low prefermance. Direction classification should perform nearly random level because temporal order is collapsed. |

### Experiment operation

#### 1.`control_T4_12pt_clean`

```PowerShell
python train_g1.py `
  --data "..\EMS-YOLO-toyproblem\self-made-dataset\generated\toy_motion_12pt_clean\control.yaml" `
  --cfg models/resnet10.yaml `
  --img 320 `
  --batch-size 8 `
  --epochs 20 `
  --device cpu `
  --workers 0 `
  --name control_T4_12pt_clean `
  -T 4
```

result path:

``g1-resnet\runs\train\control_T4_12pt_clean``

#### 2.`temporal_only_T4_12pt_clean`

```PowerShell
python train_g1.py `
  --data "..\EMS-YOLO-toyproblem\self-made-dataset\generated\toy_motion_12pt_clean\temporal_only.yaml" `
  --cfg models/resnet10.yaml `
  --img 320 `
  --batch-size 8 `
  --epochs 20 `
  --device cpu `
  --workers 0 `
  --name temporal_only_T4_12pt_clean `
  -T 4
```

result path:

``g1-resnet\runs\train\temporal_only_T4_12pt_clean``

#### 3.`temporal_only_T1_12pt_clean`

```PowerShell
python train_g1.py `
  --data "..\EMS-YOLO-toyproblem\self-made-dataset\generated\toy_motion_12pt_clean\temporal_only.yaml" `
  --cfg models/resnet10.yaml `
  --img 320 `
  --batch-size 8 `
  --epochs 20 `
  --device 0 `
  --workers 0 `
  --name temporal_only_T1_12pt_clean `
  -T 1
```

result path:

``g1-resnet\runs\train\temporal_only_T1_12pt_clean``

**If you want to use GPU acceleration, change `--device cpu` to `--device 0`.**

## Final results and conclusion

### The final dataset

``self-made-dataset/generated/toy_motion_12pt_clean``

### Output record

#### `control_T4_12pt_clean`

```text
===== control_T4_12pt_clean =====
epochs = 19
final:

epoch                : 18
metrics/precision    : 1
metrics/recall       : 0.97836
metrics/mAP_0.5      : 0.98996
metrics/mAP_0.5:0.95 : 0.71676

best mAP@0.5:

epoch                : 18
metrics/precision    : 1
metrics/recall       : 0.97836
metrics/mAP_0.5      : 0.98996
metrics/mAP_0.5:0.95 : 0.71676
```

#### `temporal_only_T4_12pt_clean`

```text
===== temporal_only_T4_12pt_clean =====
epochs = 20
final:

epoch                : 19
metrics/precision    : 0.99991
metrics/recall       : 0.99891
metrics/mAP_0.5      : 0.995
metrics/mAP_0.5:0.95 : 0.73921

best mAP@0.5:

epoch                : 17
metrics/precision    : 0.99902
metrics/recall       : 0.99996
metrics/mAP_0.5      : 0.995
metrics/mAP_0.5:0.95 : 0.74181
```

#### `temporal_only_T1_12pt_clean`

```text
===== temporal_only_T1_12pt_clean =====
epochs = 20
final:

epoch                : 19
metrics/precision    : 0.49994
metrics/recall       : 1
metrics/mAP_0.5      : 0.52258
metrics/mAP_0.5:0.95 : 0.41138

best mAP@0.5:

epoch                : 17
metrics/precision    : 0.49964
metrics/recall       : 1
metrics/mAP_0.5      : 0.53153
metrics/mAP_0.5:0.95 : 0.39506
```

### Result table

| Experiment                      | Input T | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Interpretation                                                                                                                                                       |
| ------------------------------- | ------: | --------: | -----: | ------: | -----------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `control_T4_12pt_clean`       |       4 |     1.000 |  0.978 |   0.990 |        0.717 | The current data format, bounding box design, labels, and training pipeline are learnable.                                                                           |
| `temporal_only_T4_12pt_clean` |       4 |     1.000 |  0.999 |   0.995 |        0.739 | Preserving temporal ordered event bins, the model can complete the temporal-only orientation detection task.                                                         |
| `temporal_only_T1_12pt_clean` |       1 |     0.500 |  1.000 |   0.532 |        0.395 | After folding the temporal order, the category classification is significantly degraded; the precision is close to the level of randomness in binary classification. |

### Conclusion

This toy problem supports the hypthesis：

> EMS-YOLO can use temporal order event-bin to detect the object motion orientation.

1. `control_T4_12pt_clean` achieves `mAP@0.5 = 0.98996`. This indicates that the current data format, bounding box design, labels, and training pipeline are learnable.
2. `temporal_only_T4_12pt_clean` achieves `mAP@0.5 = 0.995`. This demonstrates that EMS-YOLO can complete the detection task when only the temporal order is reliable category clue.
3. `temporal_only_T1_12pt_clean` decreases to final `mAP@0.5 = 0.52258`, best `mAP@0.5 = 0.53153`, and precision is only about `0.5`. This indicates that after the temporal order information is folded, the motion orientation category classification becomes nearly random. mAP is not 0 because the bounding box of the entire trajectory is still easy to locate. The key degradation is reflected in the ability to distinguish between categories, especially the precision, which is close to 0.5.

Therefore, we can have the conclusion that the self made dataset realized the goal.
