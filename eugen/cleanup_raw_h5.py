#!/usr/bin/env python3
import os
import glob
import sys

def main():
    print("=== Gen1 Raw H5 Dataset Cleanup & test.txt Generation ===")
    
    scratch_user = os.environ.get("USER", "ebulboaca")
    datasets_base = f"/scratch/{scratch_user}/datasets"
    raw_dir = os.path.join(datasets_base, "gen1_processed")
    processed_dir = os.path.join(datasets_base, "gen1_processed_t5")
    
    print(f"Raw Dataset directory: {raw_dir}")
    print(f"Preprocessed Dataset directory: {processed_dir}")
    
    if not os.path.exists(raw_dir):
        print(f"[ERROR] Raw Gen1 dataset not found at {raw_dir}")
        sys.exit(1)
        
    # 1. Delete all raw .h5 files to free up space (approx 1.7 TB)
    print("\nStarting cleanup of raw .h5 files...")
    splits = ['train', 'val', 'test']
    freed_files = 0
    
    for split in splits:
        split_path = os.path.join(raw_dir, split)
        if not os.path.exists(split_path):
            continue
            
        h5_files = glob.glob(os.path.join(split_path, "*.h5"))
        print(f"  Split '{split}': found {len(h5_files)} .h5 files to delete.")
        for f in h5_files:
            try:
                os.remove(f)
                freed_files += 1
            except Exception as e:
                print(f"  Error deleting {f}: {e}")
                
    print(f"[SUCCESS] Cleaned up {freed_files} raw .h5 files from scratch.")
    
    # 2. Try writing test.txt now that space has been freed
    print("\nAttempting to generate missing test.txt...")
    test_split_dst = os.path.join(processed_dir, "test")
    test_txt_path = os.path.join(processed_dir, "test.txt")
    
    if not os.path.exists(test_split_dst):
        print(f"[ERROR] Preprocessed test directory {test_split_dst} does not exist!")
        sys.exit(1)
        
    all_npy_files = sorted(glob.glob(os.path.join(test_split_dst, "img_*.npy")))
    print(f"  Found {len(all_npy_files)} preprocessed image files in {test_split_dst}")
    
    if len(all_npy_files) == 0:
        print("[ERROR] No preprocessed image files found in test split!")
        sys.exit(1)
        
    try:
        with open(test_txt_path, 'w') as f:
            for filepath in all_npy_files:
                f.write(f"{filepath}\n")
        print(f"[SUCCESS] Successfully generated {test_txt_path} containing {len(all_npy_files)} files.")
    except Exception as e:
        print(f"[ERROR] Failed to write {test_txt_path}: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
