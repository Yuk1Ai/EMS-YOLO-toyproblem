#!/bin/bash
#SBATCH --job-name=preprocess-gen1
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --output=slurm_logs/preprocess-gen1-%j.out

# Load software modules
module load 2024r1
module load python

# Add virtual environment packages to Python path
export PYTHONPATH="/home/ebulboaca/EMS-YOLO/.venv/lib/python3.10/site-packages:$PYTHONPATH"

# Preprocess Gen1 dataset
# Input: /scratch/ebulboaca/datasets/gen1_processed (which contains raw H5 + bbox.npy)
# Output: /scratch/ebulboaca/datasets/gen1_processed_t5 (which will contain the preprocessed frame tensors)
python3 eugen/preprocess_gen1_h5.py \
  --path /scratch/ebulboaca/datasets/gen1_processed \
  --outpath /scratch/ebulboaca/datasets/gen1_processed_t5 \
  -T 5

