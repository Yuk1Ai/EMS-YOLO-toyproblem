#!/usr/bin/env python3
import os
import sys
import h5py
import numpy as np
import cv2

def test_single():
    print("=== Testing Preprocessing on a Single H5 File ===")
    
    h5_path = "/scratch/ebulboaca/datasets/gen1_processed/train/compiegne_17-04-04_11-00-13_cut_10_td_61500000_121500000.h5"
    bbox_path = "/scratch/ebulboaca/datasets/gen1_processed/train/compiegne_17-04-04_11-00-13_cut_10_td_61500000_121500000_bbox.npy"
    out_dir = "/scratch/ebulboaca/datasets/test_single_out"
    
    if not os.path.exists(h5_path) or not os.path.exists(bbox_path):
        print("[ERROR] Test inputs missing on scratch. Please check paths.")
        sys.exit(1)
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Load boxes and group
    boxes = np.load(bbox_path)
    unique_ts = np.unique(boxes['ts'])
    print(f"Loaded {len(boxes)} boxes with {len(unique_ts)} unique timestamps.")
    
    # Pick the first timestamp for the test
    target_ts = unique_ts[0]
    ts_boxes = boxes[boxes['ts'] == target_ts]
    
    bin_idx = int(target_ts // 50000)
    T = 5
    print(f"Testing timestamp {target_ts} us -> maps to bin index {bin_idx}")
    
    img_tensor = np.zeros((T, 320, 320, 3), dtype=np.uint8)
    
    with h5py.File(h5_path, 'r') as h5_file:
        data_ds = h5_file['data']
        for step in range(T):
            target_bin = bin_idx - T + 1 + step
            if target_bin < 0:
                frame_3ch = 127 * np.ones((240, 304, 3), dtype=np.uint8)
            else:
                bin_data = data_ds[target_bin]
                shape = bin_data.shape
                
                # Transpose if needed
                if len(shape) == 3 and 240 in shape and 304 in shape:
                    idx_240 = shape.index(240)
                    idx_304 = shape.index(304)
                    idx_c = [i for i in range(3) if i != idx_240 and i != idx_304][0]
                    bin_data = bin_data.transpose(idx_c, idx_240, idx_304)
                else:
                    raise ValueError(f"Unexpected shape: {shape}")
                
                # Extract positive (even) and negative (odd) channels
                pos_channels = bin_data[0::2]
                neg_channels = bin_data[1::2]
                
                pos_mask = np.any(pos_channels > 0, axis=0)
                neg_mask = np.any(neg_channels > 0, axis=0)
                
                frame_3ch = 127 * np.ones((240, 304, 3), dtype=np.uint8)
                frame_3ch[neg_mask] = 0
                frame_3ch[pos_mask] = 255
                
            img_tensor[step] = cv2.resize(frame_3ch, (320, 320))
            
    # Verify shape
    print(f"Generated frame tensor shape: {img_tensor.shape} (Expected: (5, 320, 320, 3))")
    print(f"Generated frame tensor dtype: {img_tensor.dtype} (Expected: uint8)")
    
    assert img_tensor.shape == (5, 320, 320, 3), "Shape mismatch!"
    assert img_tensor.dtype == np.uint8, "Dtype mismatch!"
    
    # Save the npy
    npy_out = os.path.join(out_dir, "test_slice.npy")
    np.save(npy_out, img_tensor)
    print(f"Successfully saved test npy slice to: {npy_out}")
    
    # Generate labels
    yolo_labels = []
    for box in ts_boxes:
        class_id = int(box['class_id'])
        x = float(box['x'])
        y = float(box['y'])
        w = float(box['w'])
        h = float(box['h'])
        
        center_x = (x + w / 2.0) / 304.0
        center_y = (y + h / 2.0) / 240.0
        norm_w = w / 304.0
        norm_h = h / 240.0
        
        yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
        
    txt_out = os.path.join(out_dir, "test_slice.txt")
    with open(txt_out, 'w') as f:
        f.write("\n".join(yolo_labels) + "\n")
        
    print(f"Successfully saved test label text to: {txt_out}")
    print("Labels content:")
    for label in yolo_labels:
        print(f"  {label}")
        
    print("\n=== Single-file preprocessing test PASSED successfully! ===")

if __name__ == "__main__":
    test_single()
