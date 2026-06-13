#!/bin/bash
#
# Assumes you are on TU Delft Wi-Fi or connected via EduVPN (no bastion hop).
#
# Usage:
#   ./sync_to_delftblue.sh <netid>

set -euo pipefail

NETID="${1:-${NETID:-ebulboaca}}"
if [[ -z "$NETID" ]]; then
  echo "Usage: $0 [netid]   (defaults to ebulboaca)" >&2
  exit 1
fi

REMOTE_HOST="login.delftblue.tudelft.nl"
REMOTE_PATH="~/EMS-YOLO/"

# ServerAlive keeps the connection from dying mid-transfer on flaky Wi-Fi.
SSH_CMD="ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3"

rsync -avzh --partial --progress \
  -e "$SSH_CMD" \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='.mypy_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.pytest_cache/' \
  --exclude='/wandb/' \
  --exclude='slurm_logs/' \
  --exclude='logs/' \
  ./ "${NETID}@${REMOTE_HOST}:${REMOTE_PATH}"
