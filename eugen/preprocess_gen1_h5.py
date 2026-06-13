#!/usr/bin/env python3
import os
import sys
import h5py
import numpy as np
import cv2
import argparse
import glob
from tqdm import tqdm
from multiprocessing import Pool

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess pre-binned Gen1 H5 dataset for EMS-YOLO")
    parser.add_argument("--path", type=str, required=True, help="Path to university pre-processed Gen1 dataset (H5 + bbox.npy)")
    parser.add_argument("--outpath", type=str, required=True, help="Path to save processed dataset slices")
    parser.add_argument("-T", type=int, default=5, help="Number of time-steps")
    return parser.parse_args()

def process_sequence(args):
    bbox_name, split_src, split_dst, T = args
    base_name = bbox_name[:-9] # strip '_bbox.npy'
    h5_name = base_name + '.h5'
    
    bbox_filepath = os.path.join(split_src, bbox_name)
    h5_filepath = os.path.join(split_src, h5_name)
    
    if not os.path.exists(h5_filepath):
        # Retry looking for td suffix
        h5_name = base_name + '_td.h5'
        h5_filepath = os.path.join(split_src, h5_name)
        if not os.path.exists(h5_filepath):
            return 0
            
    # Load boxes
    try:
        boxes = np.load(bbox_filepath)
    except Exception as e:
        print(f"Error loading {bbox_filepath}: {e}")
        return 0
        
    if len(boxes) == 0:
        return 0
        
    unique_ts = np.unique(boxes['ts'])
    
    # 1. Resume check: Check if all expected files already exist
    all_exist = True
    for ts in unique_ts:
        out_img_path = os.path.join(split_dst, f"img_{base_name}_{ts}.npy")
        out_lbl_path = os.path.join(split_dst, f"img_{base_name}_{ts}.txt")
        if not os.path.exists(out_img_path) or not os.path.exists(out_lbl_path):
            all_exist = False
            break
            
    if all_exist:
        return len(unique_ts) # Already processed this sequence
        
    # 2. Slice and process missing frames
    samples_created = 0
    try:
        with h5py.File(h5_filepath, 'r') as h5_file:
            data_arr = h5_file['data'][:] # Load entire dataset into RAM to avoid disk seek bottleneck
            
            for ts in unique_ts:
                out_img_path = os.path.join(split_dst, f"img_{base_name}_{ts}.npy")
                out_lbl_path = os.path.join(split_dst, f"img_{base_name}_{ts}.txt")
                
                # Skip individual frame if already exists
                if os.path.exists(out_img_path) and os.path.exists(out_lbl_path):
                    samples_created += 1
                    continue
                    
                ts_boxes = boxes[boxes['ts'] == ts]
                
                # Convert microsecond timestamp to H5 bin index (50ms bins)
                bin_idx = int(ts // 50000)
                
                if bin_idx >= data_arr.shape[0]:
                    continue
                    
                # Construct event frame tensor of shape (T, 320, 320, 3)
                img_tensor = np.zeros((T, 320, 320, 3), dtype=np.uint8)
                
                for step in range(T):
                    target_bin = bin_idx - T + 1 + step
                    
                    if target_bin < 0:
                        # Pad with background frame (127 gray)
                        frame_3ch = 127 * np.ones((240, 304, 3), dtype=np.uint8)
                    else:
                        # Load bin
                        bin_data = data_arr[target_bin]
                        
                        # Ensure bin_data is transposed to shape (channels, 240, 304) dynamically
                        shape = bin_data.shape
                        if len(shape) == 3 and 240 in shape and 304 in shape:
                            idx_240 = shape.index(240)
                            idx_304 = shape.index(304)
                            idx_c = [i for i in range(3) if i != idx_240 and i != idx_304][0]
                            bin_data = bin_data.transpose(idx_c, idx_240, idx_304)
                        else:
                            raise ValueError(f"Unexpected bin_data shape: {shape} in {h5_filepath}")
                            
                        # Extract positive (even) and negative (odd) channels
                        pos_channels = bin_data[0::2]
                        neg_channels = bin_data[1::2]
                        
                        pos_mask = np.any(pos_channels > 0, axis=0)
                        neg_mask = np.any(neg_channels > 0, axis=0)
                        
                        # Initialize background
                        frame_3ch = 127 * np.ones((240, 304, 3), dtype=np.uint8)
                        frame_3ch[neg_mask] = 0
                        frame_3ch[pos_mask] = 255
                        
                    # Resize to 320x320
                    img_tensor[step] = cv2.resize(frame_3ch, (320, 320))
                    
                # Normalize bounding box coordinates for YOLO
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
                    
                    # Clip to bounds [0, 1]
                    center_x = min(max(center_x, 0.0), 1.0)
                    center_y = min(max(center_y, 0.0), 1.0)
                    norm_w = min(max(norm_w, 0.0), 1.0)
                    norm_h = min(max(norm_h, 0.0), 1.0)
                    
                    if norm_w > 0 and norm_h > 0:
                        yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
                        
                if len(yolo_labels) == 0:
                    continue # Skip frames without valid bounding boxes
                    
                np.save(out_img_path, img_tensor)
                with open(out_lbl_path, 'w') as lf:
                    lf.write("\n".join(yolo_labels) + "\n")
                    
                samples_created += 1
    except Exception as e:
        print(f"\n[ERROR] Failed to process sequence {base_name}: {e}")
        return 0
        
    return samples_created

def main():
    args = parse_args()
    raw_path = args.path
    out_path = args.outpath
    T = args.T
    
    print(f"=== Custom Parallelized & Resumeable Preprocessing Gen1 H5 dataset for T={T} ===")
    print(f"Source: {raw_path}")
    print(f"Destination: {out_path}")
    
    splits = ['train', 'val', 'test']
    os.makedirs(out_path, exist_ok=True)
    
    # Get worker pool size from Slurm or default to 8
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    print(f"Using {num_workers} parallel workers.")
    
    for split in splits:
        split_src = os.path.join(raw_path, split)
        split_dst = os.path.join(out_path, split)
        
        if not os.path.exists(split_src):
            print(f"Warning: split directory {split_src} not found. Skipping.")
            continue
            
        os.makedirs(split_dst, exist_ok=True)
        txt_path = os.path.join(out_path, f"{split}.txt")
        
        print(f"\nProcessing split: {split}")
        
        # Find all bbox files
        bbox_files = sorted([f for f in os.listdir(split_src) if f.endswith('_bbox.npy')])
        
        # Build arguments list for worker pool
        args_list = [(bbox_name, split_src, split_dst, T) for bbox_name in bbox_files]
        
        # Run multiprocessing Pool
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_sequence, args_list), total=len(bbox_files), desc=f"Split {split}"))
            
        total_samples = sum(results)
        print(f"Split {split} completed. Processed/verified {total_samples} samples.")
        
        # Write index split.txt file listing all absolute paths of existing .npy files
        print(f"Writing index file: {txt_path}")
        all_npy_files = sorted(glob.glob(os.path.join(split_dst, "img_*.npy")))
        with open(txt_path, 'w') as f:
            for filepath in all_npy_files:
                f.write(f"{filepath}\n")
        print(f"Successfully generated {txt_path} containing {len(all_npy_files)} files.")

if __name__ == "__main__":
    main()
