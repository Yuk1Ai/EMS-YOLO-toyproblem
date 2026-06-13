#!/usr/bin/env python3
import os
import glob
import h5py
from collections import Counter

def main():
    print("=== Scanning all Gen1 H5 Files to Find All Unique Shapes ===")
    
    datasets_base = "/scratch/ebulboaca/datasets/gen1_processed"
    splits = ['train', 'val', 'test']
    
    if not os.path.exists(datasets_base):
        print(f"[ERROR] Gen1 processed folder not found at {datasets_base}")
        return

    shape_counter = Counter()
    corrupt_files = []
    
    for split in splits:
        split_dir = os.path.join(datasets_base, split)
        if not os.path.exists(split_dir):
            print(f"Split folder {split_dir} does not exist. Skipping.")
            continue
            
        h5_files = glob.glob(os.path.join(split_dir, "*.h5"))
        print(f"\nScanning split '{split}': found {len(h5_files)} H5 files...")
        
        for filepath in h5_files:
            try:
                with h5py.File(filepath, 'r') as f:
                    if 'data' not in f:
                        print(f"  [MISSING KEY] 'data' not found in {os.path.basename(filepath)}")
                        shape_counter[("missing_data",)] += 1
                        continue
                    
                    shape = f['data'].shape
                    shape_counter[shape] += 1
            except Exception as e:
                print(f"  [CORRUPT] Failed to open {os.path.basename(filepath)}: {e}")
                corrupt_files.append(filepath)
                
    print("\n=== Shape Scan Results ===")
    for shape, count in shape_counter.items():
        print(f"Shape: {shape} -> {count} files")
        
    if corrupt_files:
        print(f"\nFound {len(corrupt_files)} corrupt files:")
        for cf in corrupt_files:
            print(f"  {cf}")
            
    print("\n=== Scan Completed ===")

if __name__ == "__main__":
    main()
