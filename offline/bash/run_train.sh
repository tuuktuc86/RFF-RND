#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$OFFLINE_DIR"
python train_offline_rl.py config/train_default_config.yaml
