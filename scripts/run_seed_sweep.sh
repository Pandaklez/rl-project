#!/bin/bash
# Three conditions x three seeds, matched configuration, then evaluation.
#
# Parallelism: each run is capped to 3 BLAS/OMP threads rather than letting torch
# grab all 32 cores. The networks are small MLPs that scale badly past a few
# threads -- the earlier serial runs sat at ~650% CPU for one run -- so 9 runs at
# 3 threads uses 27 of 32 cores and finishes far sooner than 9 runs in sequence.
#
# Pose-image logging is off (--viz_interval 0). It decodes video per figure, and
# nine processes competing for the same files is not worth it; PA-MPJPE from
# src/evaluate.py is the metric this sweep exists to produce, and the training
# curves already live in checkpoints/biasfix.
set -u
PY=/home/annkle/miniconda/envs/smplerx/bin/python
ROOT=/home/annkle/rl-project
cd "$ROOT" || exit 1

export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3

SEEDS="42 43 44"
OUT=checkpoints/sweep
mkdir -p "$OUT" /tmp/sweep

flags_for () {
  case "$1" in
    frozen)  echo "--trans_mode none" ;;
    uv)      echo "--trans_mode uv" ;;
    notrans) echo "--trans_mode none --no_state_trans" ;;
  esac
}

echo "=== PHASE 1: training, 9 runs in parallel  $(date) ==="
for seed in $SEEDS; do
  for cond in frozen uv notrans; do
    tag="${cond}_s${seed}"
    $PY -u -m src.train \
        --h5_path data/processed_movi.h5 --reward_mode reproj \
        $(flags_for "$cond") \
        --rollouts 3200 --mini_batches 6 --total_updates 40 \
        --out_dir "$OUT/$tag" --log_interval 400 --viz_interval 0 \
        --device cpu --seed "$seed" > "/tmp/sweep/train_${tag}.log" 2>&1 &
  done
done
wait
echo "=== PHASE 1 done  $(date) ==="

echo "=== PHASE 2: baseline (deterministic, one run) ==="
$PY -u -m src.evaluate --processed_h5 data/processed_movi.h5 \
    --lifted_h5 lifted_movi_part1_upd2.h5 --split test \
    --joint_set j14 --betas gt --device cpu \
    --dump_scores /tmp/sweep/scores_baseline.json > /tmp/sweep/eval_baseline.log 2>&1
echo "=== baseline done  $(date) ==="

echo "=== PHASE 3: evaluation, 9 runs in parallel  $(date) ==="
for seed in $SEEDS; do
  for cond in frozen uv notrans; do
    tag="${cond}_s${seed}"
    ck="$OUT/$tag/actor_final.pt"
    if [ ! -f "$ck" ]; then echo "!! missing $ck"; continue; fi
    $PY -u -m src.evaluate --processed_h5 data/processed_movi.h5 \
        --checkpoint "$ck" --split test --joint_set j14 --betas gt --device cpu \
        --dump_scores "/tmp/sweep/scores_${tag}.json" \
        > "/tmp/sweep/eval_${tag}.log" 2>&1 &
  done
done
wait
echo "=== PHASE 3 done  $(date) ==="
echo "SWEEP COMPLETE"
