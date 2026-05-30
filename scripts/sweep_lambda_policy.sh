#!/usr/bin/env bash
# Sweep lambda_policy for the per-element advantage actor-critic.
#
# Skips 0.3 since we already have that result (test AUC 0.7838).
# Total runtime: ~75 min on this hardware (3 runs × 25 min).
#
# Usage from project root:
#   bash scripts/sweep_lambda_policy.sh

set -e

PYTHON=/c/Users/fadyh/miniconda3/envs/torch_cuda/python.exe
LOGDIR=logs/sweep_lambda_policy
mkdir -p "$LOGDIR"

for LP in 0.1 0.15 0.2; do
    echo "=== $(date '+%H:%M:%S') :: lambda_policy=$LP ==="
    $PYTHON train.py \
        --data yelp --model CARE \
        --use-capn --use-actor-critic \
        --gnn-warmup-epochs 5 --lambda-policy-ramp-epochs 5 \
        --llm-priors-file data/llm_priors/yelp_priors.json \
        --text-enrichment gate --text-state-enrichment \
        --seed 72 \
        --lambda-policy "$LP" \
        > "$LOGDIR/lp_${LP}.log" 2>&1
    echo "    done -> $LOGDIR/lp_${LP}.log"
done

echo
echo "=== Sweep complete. Best-checkpoint AUCs: ==="
for LP in 0.1 0.15 0.2; do
    LOG="$LOGDIR/lp_${LP}.log"
    BEST_AUC=$(grep "New best model saved" "$LOG" | tail -1 | grep -oE 'AUC: 0\.[0-9]+' | grep -oE '0\.[0-9]+')
    echo "  lambda_policy=$LP  best val AUC=$BEST_AUC  ($LOG)"
done
