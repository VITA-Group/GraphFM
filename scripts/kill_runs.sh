#!/usr/bin/env bash
set -euo pipefail

pkill -f "python scripts/run_experiment.py" || true
pkill -f "bash train.sh" || true
