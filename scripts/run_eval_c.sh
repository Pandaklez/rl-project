#!/bin/bash
# Evaluations for the (C) column of report.md, from checkpoints/gail_c.
#
# Two metrics, matching the two (B) tables so (C) is comparable:
#   1. PA-MPJPE on test, via src/evaluate.py --dump_scores. Named
#      gail_<cond>_s<seed>.json because scripts/sweep_stats.py's GAIL_VARIANTS
#      looks for exactly that in --scores_dir.
#   2. Held-out reprojection improvement, via scripts/heldout_eval.py, with the
#      same --split test --n_clips 400 --clip_seed 42 as the (B) row so the
#      clips are identical.
#
# The rollout/config/diagnostics tables read TensorBoard event files that
# already exist under checkpoints/gail_c/*/, so nothing needs re-running there.
set -u
PY=/home/annkle/miniconda/envs/smplerx/bin/python
ROOT=/home/annkle/rl-project
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

LOG=logs/eval_c
mkdir -p eval_scores "$LOG"

echo "=== (C) PA-MPJPE dumps, start $(date) ==="
for cond in feet_out feet_in; do
  for seed in 42 43 44; do
    $PY -u -m src.evaluate \
        --processed_h5 data/processed_movi.h5 \
        --lifted_h5 lifted_movi_part1_upd2.h5 \
        --joint_set j14 --split test --device cpu \
        --checkpoint "checkpoints/gail_c/${cond}_s${seed}/actor_final.pt" \
        --dump_scores "eval_scores/gail_${cond}_s${seed}.json" \
        > "$LOG/pampjpe_${cond}_s${seed}.log" 2>&1 &
  done
done
wait
echo "=== (C) PA-MPJPE dumps, done $(date) ==="

echo "=== (C) held-out reprojection, start $(date) ==="
$PY -u scripts/heldout_eval.py \
    --sweep_dir checkpoints/gail_c \
    --runs feet_out_s42 feet_out_s43 feet_out_s44 \
           feet_in_s42 feet_in_s43 feet_in_s44 \
    --split test --n_clips 400 --n_frames 12 --clip_seed 42 \
    --dump eval_scores/heldout_test_gail_c.json \
    > "$LOG/heldout.log" 2>&1
echo "=== (C) held-out reprojection, done $(date) ==="

echo "=== aggregate ==="
$PY scripts/sweep_stats.py --table pampjpe --scores_dir eval_scores
