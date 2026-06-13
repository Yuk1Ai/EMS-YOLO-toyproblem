#!/bin/bash
#SBATCH --job-name=check-quota
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=slurm_logs/check-quota-%j.out

echo "=== Disk Space Usage ==="
df -h /scratch
df -h /home/ebulboaca

echo "=== Inode (File Count) Usage ==="
df -i /scratch
df -i /home/ebulboaca

echo "=== Quota Info (if available) ==="
quota -s || echo "quota command not available"
lfs quota -u ebulboaca /scratch || echo "lfs quota not available"
myquota || echo "myquota not available"

echo "=== Scratch Subdirectory Sizes ==="
du -h --max-depth=2 /scratch/ebulboaca || echo "Failed to get scratch sizes"

echo "=== File Count in Scratch ==="
find /scratch/ebulboaca -type f | wc -l || echo "Failed to get file count"
