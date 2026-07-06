#!/usr/bin/env bash
# Re-run the 4-row YelpChi table on the v3 backbone (feat-encoder + ego-sep +
# cosine, 80 epochs), 5 seeds, val-selected checkpoints, Haiku text scores.
# Resumable: a row whose results JSON already exists is skipped.
set -u
cd "$(dirname "$0")/.."

PY="C:/Users/fadyh/miniconda3/envs/torch_cuda/python.exe"
SEEDS="72 73 74 75 76"
V3="--feat-encoder --ego-separation --cosine-lr --num-epochs 80 --lr-warmup-epochs 8 --test-epochs 5 --patience 12"
CAPN="--use-capn --use-actor-critic --gnn-warmup-epochs 5 --lambda-policy-ramp-epochs 5 --llm-priors-file data/llm_priors/yelp_priors.json --lambda-policy 0.15"

run_row() {
    local tag="$1"; shift
    if [ -f "results/experiments/${tag}.json" ]; then
        echo "[skip] ${tag} already done"
        return
    fi
    echo "[$(date +%H:%M:%S)] starting ${tag}"
    "$PY" scripts/run_yelp_experiment.py --tag "$tag" --seeds $SEEDS -- "$@"
}

# cheap rows first so partial results arrive early
run_row v3_row1_5seed --text-enrichment none $V3
run_row v3_row3_5seed --text-enrichment gate $V3
run_row v3_row2_5seed $CAPN $V3
run_row v3_row4_5seed $CAPN --text-enrichment gate --text-state-enrichment $V3
echo "[$(date +%H:%M:%S)] all rows done"
