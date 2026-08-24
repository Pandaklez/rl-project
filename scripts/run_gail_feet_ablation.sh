#!/bin/bash
# Experiment (C): does hiding the feet from the discriminator matter?
#
# Two conditions x three seeds, everything else matched to the (B1) frozen arm
# of checkpoints/sweep so (C) is comparable to the (B) table in report.md:
# 40 updates x 3,200 = 128,000 timesteps, reproj reward, translation frozen,
# policy on 100% of train (demo_frac=0).
#
#   feet_in   discriminator sees 21 joints (63 dims), excluding global_orient
#   feet_out  discriminator sees 19 joints (57 dims), also excluding the feet
#
# Why the feet are suspect: SMPLer-X predicts joints 10/11 near-constant
# (sd ~0.013) while MoVi GT articulates them (sd ~0.095), so they separate real
# from fake without saying anything about pose plausibility -- the same shape as
# the finger leak. Excluding them cuts LDA separability from d'=8.3 to d'=5.9.
#
# Three seeds because this repo has already retracted one single-seed result
# (see report.md, "(B3)'s apparent advantage was seed 42").
#
# Parallelism: 6 runs x 3 threads = 18 of 32 cores; the nets are small MLPs that
# scale badly past a few threads.
set -u
PY=/home/annkle/miniconda/envs/smplerx/bin/python
ROOT=/home/annkle/rl-project
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3

OUT=checkpoints/gail_c
LOG=/tmp/gail_c
mkdir -p "$OUT" "$LOG"

echo "=== experiment C feet ablation, start $(date) ==="
for seed in 42 43 44; do
  for cond in feet_in feet_out; do
    if [ "$cond" = "feet_in" ]; then EXCL="0"; else EXCL="0 10 11"; fi
    tag="${cond}_s${seed}"
    $PY -u -m src.gail_train \
        --h5_path data/processed_movi.h5 --reward_mode reproj \
        --trans_mode none \
        --rollouts 3200 --mini_batches 6 --total_updates 40 \
        --w_gail 0.5 --disc_grad_penalty 50.0 \
        --disc_exclude_joints $EXCL \
        --demo_frac 0.0 --demo_probe_frac 0.05 \
        --out_dir "$OUT/$tag" --log_interval 400 \
        --viz_interval 5 --viz_mode image \
        --device cpu --seed "$seed" > "$LOG/${tag}.log" 2>&1 &
  done
done
wait
echo "=== experiment C feet ablation, done $(date) ==="
