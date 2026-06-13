#!/usr/bin/env python3
import os
import glob
import sys

def main():
    print("=== Gen1 Dataset Diagnostic & Repair ===")
    
    scratch_user = os.environ.get("USER", "ebulboaca")
    datasets_base = f"/scratch/{scratch_user}/datasets"
    raw_dir = os.path.join(datasets_base, "gen1_processed")
    processed_dir = os.path.join(datasets_base, "gen1_processed_t5")
    
    print(f"Raw Dataset directory: {raw_dir}")
    print(f"Preprocessed Dataset directory: {processed_dir}")
    
    # 1. Check Raw Dataset
    if not os.path.exists(raw_dir):
        print(f"\n[ERROR] Raw Gen1 dataset not found at {raw_dir}")
        print("Please check your download paths.")
        sys.exit(1)
        
    print("\nChecking raw files...")
    raw_splits = ['train', 'val', 'test']
    raw_ok = True
    for split in raw_splits:
        split_path = os.path.join(raw_dir, split)
        if not os.path.exists(split_path):
            print(f"  [MISSING] Raw split folder: {split_path}")
            raw_ok = False
            continue
        h5_files = glob.glob(os.path.join(split_path, "*.h5"))
        bbox_files = glob.glob(os.path.join(split_path, "*_bbox.npy"))
        print(f"  Split '{split}': found {len(h5_files)} event files (.h5) and {len(bbox_files)} label files (_bbox.npy)")
        if len(h5_files) == 0 or len(bbox_files) == 0:
            raw_ok = False
            
    if not raw_ok:
        print("[WARNING] Raw dataset directory structure seems incomplete or corrupted.")
    else:
        print("[SUCCESS] Raw dataset files are present and verified.")
        
    # 2. Check Preprocessed Dataset
    if not os.path.exists(processed_dir):
        print(f"\n[INFO] Preprocessed dataset directory {processed_dir} does not exist yet.")
        print("You must preprocess the raw dataset first before training.")
        print("Please run the preprocessing Slurm batch job using:")
        print("  sbatch eugen/preprocess_gen1.sbatch")
        sys.exit(0)
        
    print("\nChecking preprocessed files...")
    splits = ['train', 'val', 'test']
    all_ok = True
    for split in splits:
        split_dir = os.path.join(processed_dir, split)
        txt_file = os.path.join(processed_dir, f"{split}.txt")
        
        print(f"\n--- Processed Split: {split} ---")
        if not os.path.exists(split_dir):
            print(f"  [MISSING] Split directory {split_dir} does not exist!")
            all_ok = False
            continue
            
        npy_files = glob.glob(os.path.join(split_dir, "img_*.npy"))
        label_txt_files = glob.glob(os.path.join(split_dir, "img_*.txt"))
        
        print(f"  Found {len(npy_files)} preprocessed image frames (img_*.npy)")
        print(f"  Found {len(label_txt_files)} preprocessed label files (img_*.txt)")
        
        if len(npy_files) == 0:
            print(f"  [WARNING] No preprocessed frame files in {split_dir}!")
            all_ok = False
            continue
            
        # Check if list txt file exists and is correct
        needs_txt_write = False
        if os.path.exists(txt_file):
            print(f"  List file {txt_file} exists.")
            with open(txt_file, 'r') as f:
                lines = f.read().splitlines()
            print(f"  List file contains {len(lines)} paths.")
            if len(lines) == 0 or not os.path.exists(lines[0]):
                print("  [WARNING] Paths inside the index file are invalid or empty. Repairing...")
                needs_txt_write = True
        else:
            print(f"  List file {txt_file} is MISSING. Generating...")
            needs_txt_write = True
            
        if needs_txt_write:
            npy_files.sort()
            with open(txt_file, 'w') as f:
                for path in npy_files:
                    f.write(f"{path}\n")
            print(f"  [SUCCESS] Generated {txt_file} containing {len(npy_files)} files.")
            
    if all_ok:
        print("\n=== All Gen1 Dataset checks PASSED! ===")
        print("You can now run the dry-run command:")
        print("  cd g1-resnet")
        print("  WANDB_MODE=disabled uv run python3 train_g1.py --data data/gen1.yaml --epochs 1 --batch-size 2 --device cpu")
    else:
        print("\n[INFO] Preprocessed dataset is incomplete.")
        print("Please submit the preprocessing Slurm batch job using:")
        print("  sbatch eugen/preprocess_gen1.sbatch")

if __name__ == "__main__":
    main()
