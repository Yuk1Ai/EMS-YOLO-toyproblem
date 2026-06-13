#!/usr/bin/env python3
"""
test_coco_val_batch.py
Verifies that the validation loop in g1-resnet/val.py can run a single batch
of COCO validation on the CPU without unpacking crashes.
"""

import os
import sys
from pathlib import Path
import torch

# Define workspace directories
ROOT_DIR = Path(__file__).resolve().parent.parent
g1_resnet_dir = ROOT_DIR / 'g1-resnet'

# Add g1-resnet to Python path
sys.path.insert(0, str(g1_resnet_dir))

from models.common import DetectMultiBackend
import models.yolo
import models.common
from utils.general import check_dataset, check_yaml
import val

class SingleBatchDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset

    def __len__(self):
        return 1

    def __iter__(self):
        for batch in self.dataloader:
            yield batch
            break

def test_batch_run():
    print("=== Starting 1-Batch COCO Validation Test (CPU) ===")
    
    device = torch.device("cpu")
    weights_path = ROOT_DIR / "runs/train/exp/weights/best.pt"
    data_yaml = g1_resnet_dir / "data/coco.yaml"
    
    if not weights_path.exists():
        print(f"Error: Weights file not found at {weights_path}")
        sys.exit(1)
        
    print(f"Loading weights from: {weights_path}...")
    models.yolo.time_window = 5
    models.common.time_window = 5
    
    # Load data dict
    data_dict = check_dataset(check_yaml(data_yaml))
    val_path = data_dict['val']
    
    # Create the real COCO dataloader
    from utils.datasets import create_dataloader as create_img_dataloader
    print(f"Initializing dataloader for path: {val_path}...")
    real_dataloader = create_img_dataloader(
        val_path, imgsz=640, batch_size=2, stride=32, single_cls=False,
        pad=0.5, rect=True, workers=1, prefix="val: "
    )[0]
    
    # Wrap dataloader to return only 1 batch
    test_dataloader = SingleBatchDataLoader(real_dataloader)
    print("Dataloader wrapped successfully. Running 1-batch validation loop...")
    
    try:
        # Call val.run with our custom dataloader
        results, maps, times = val.run(
            data=data_yaml,
            weights=weights_path,
            batch_size=2,
            imgsz=640,
            conf_thres=0.001,
            iou_thres=0.6,
            task='val',
            device='cpu',
            single_cls=False,
            augment=False,
            verbose=True,
            save_txt=False,
            save_hybrid=False,
            save_conf=False,
            save_json=False,
            plots=False,
            model=None,
            dataloader=test_dataloader
        )
        print("=== Test finished: NO ERRORS FOUND ===")
        print(f"Results: {results}")
        
    except Exception as e:
        print("Validation test failed:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_batch_run()
