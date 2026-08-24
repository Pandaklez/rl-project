#!/bin/bash
# Experiment (D): does a supervised term help?
#
# (B3) pose-only, plus a third reward term -- the mean squared error between the
# corrected pose and the GT pose of the *same* frame. So the reward is
#
#     reprojection  +  smoothness  +  MSE(corrected_t, gt_t)
#
# which makes (D) the supervised counterpart of (B): (A)/(B)/(C) are all
# ground-truth-free by construction, and (D) deliberately is not. If correction
# is learnable at all, the arm that is handed the answer should show it.
#
# Matched to the (B3) arm of checkpoints/sweep on everything else -- 40 updates
# x 3,200 = 128,000 timesteps, reproj reward, --trans_mode none --no_state_trans
# -- so (D) is comparable to the (B) tables in report.md. Launched with image
# pose logging on, matching how the 22-joint (B) checkpoints and the (C) runs
# were actually produced (report.md notes run_seed_sweep.sh's committed
# --viz_interval 0 is NOT how those checkpoints were made).
#
# Two weights, because one weight cannot distinguish "supervision does not help"
# from "that weight was wrong". At w_mse=1 the MSE improvement term is ~0.003
# against a base reward of ~0.01, i.e. the same order; w_mse=10 makes it
# dominant. Three seeds each, because this repo has already retracted one
# single-seed result (report.md, "(B3)'s apparent advantage was seed 42").
#
# Parallelism: 6 runs x 3 threads = 18 of 32 cores, the same footprint the (C)
# ablation ran at safely. The nets are small MLPs that scale badly past a few
# threads. NOTE it was evaluation, not training, that took this machine down --
# see run_eval_d_light.sh.
set -u
PY=/home/annkle/miniconda/envs/smplerx/bin/python
ROOT=/home/annkle/rl-project
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3

OUT=checkpoints/exp_d
LOG=logs/exp_d
mkdir -p "$OUT" "$LOG"

echo "=== experiment (D) supervised MSE, start $(date) ==="
for seed in 42 43 44; do
  for cond in mse1 mse10; do
    case "$cond" in
      mse1)  W=1.0  ;;
      mse10) W=10.0 ;;
    esac
    tag="${cond}_s${seed}"
    if [ -f "$OUT/$tag/actor_final.pt" ]; then
      echo "[skip] $tag already trained"; continue
    fi
    $PY -u -m src.train \
        --h5_path data/processed_movi.h5 --reward_mode reproj \
        --trans_mode none --no_state_trans \
        --w_mse "$W" \
        --rollouts 3200 --mini_batches 6 --total_updates 40 \
        --out_dir "$OUT/$tag" --log_interval 400 \
        --viz_interval 5 --viz_mode image \
        --device cpu --seed "$seed" > "$LOG/train_${tag}.log" 2>&1 &
  done
done
wait
echo "=== experiment (D) training done $(date) ==="

for seed in 42 43 44; do
  for cond in mse1 mse10; do
    ck="$OUT/${cond}_s${seed}/actor_final.pt"
    [ -f "$ck" ] || echo "!! MISSING $ck"
  done
done
