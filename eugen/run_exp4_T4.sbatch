#!/bin/bash
#SBATCH --job-name=exp4-T4
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=8000M
#SBATCH --output=slurm_logs/exp4-T4-%j.out

# Load software modules
module load 2024r1
module load openmpi
module load cuda/12.4
module load python

# Add virtual environment packages to Python path
export PYTHONPATH="/home/ebulboaca/EMS-YOLO/.venv/lib/python3.10/site-packages:$PYTHONPATH"
export WANDB_MODE=disabled

# Execute inside g1-resnet/ for event-based data run isolation
cd g1-resnet
python3 train_g1.py \
  --data data/gen1.yaml \
  --img 320 \
  --batch-size 16 \
  --epochs 50 \
  --weights "" \
  --device 0 \
  --project /scratch/$USER/runs/stage3 \
  --name T_4 \
  -T 4 \
  --cache
