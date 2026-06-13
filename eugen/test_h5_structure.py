#!/usr/bin/env python3
import os
import h5py
import numpy as np

def main():
    print("=== Gen1 H5 and Bbox Dataset Inspection ===")
    
    h5_path = "/scratch/ebulboaca/datasets/gen1_processed/val/can_recording_17-08-29_10-56-11_td_488500000_548500000.h5"
    bbox_path = "/scratch/ebulboaca/datasets/gen1_processed/val/can_recording_17-08-29_10-56-11_td_488500000_548500000_bbox.npy"
    
    if not os.path.exists(h5_path):
        print(f"[ERROR] H5 file not found at: {h5_path}")
        return
    if not os.path.exists(bbox_path):
        print(f"[ERROR] Bbox file not found at: {bbox_path}")
        return

    # 1. Inspect H5 File
    print(f"\nLoading H5 file: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        print("Keys:", list(f.keys()))
        if 'data' not in f:
            print("[ERROR] 'data' key not found in H5 file.")
            return
        
        data_ds = f['data']
        print("Shape:", data_ds.shape)
        print("Dtype:", data_ds.dtype)
        
        # Let's inspect some values across channels
        print("\nAnalyzing channel statistics across the entire sequence:")
        for c in range(data_ds.shape[1]):
            # Load a subset of the sequence to avoid loading the entire 1200 frames into memory
            # We will load 100 frames from the middle
            chunk = data_ds[500:600, c, :, :]
            non_zero = np.count_nonzero(chunk)
            total = chunk.size
            pct = (non_zero / total) * 100
            print(f"  Channel {c}: min={chunk.min()}, max={chunk.max()}, mean={chunk.mean():.4f}, non-zero={pct:.2f}%")
            
            # Print unique values
            unique_vals = np.unique(chunk)
            if len(unique_vals) < 10:
                print(f"    Unique values: {unique_vals}")
            else:
                print(f"    Unique values (first 10): {unique_vals[:10]}")

    # 2. Inspect Bbox File
    print(f"\nLoading Bbox file: {bbox_path}")
    bbox = np.load(bbox_path)
    print("Bbox Dtype:", bbox.dtype)
    print("Bbox Shape:", bbox.shape)
    print("First 5 bounding box samples:")
    for i in range(min(5, len(bbox))):
        print(f"  {bbox[i]}")
        
    # Check bounding box timestamps relative to H5 sequence length
    ts_min, ts_max = bbox['ts'].min(), bbox['ts'].max()
    print(f"\nTimestamps in Bbox: min={ts_min} us, max={ts_max} us")
    print(f"Duration of H5 recording is 60 seconds (60,000,000 us)")
    print(f"Let's check target bin index mapping: idx = ts // 50000")
    print("First 5 mapped indices:")
    for i in range(min(5, len(bbox))):
        ts = bbox['ts'][i]
        print(f"  ts={ts} us -> bin_idx={ts // 50000}")

if __name__ == "__main__":
    main()
