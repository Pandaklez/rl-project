#!/bin/bash
# Experiment (E): the plain supervised regression benchmark.
#
# Direct-regression counterpart to (D) (`scripts/run_exp_d.sh`): (D) adds an
# MSE-to-GT term *inside a PPO reward*; (E) instead trains a plain per-frame
# MLP to minimise that same error by gradient descent -- no PPO, no rollout,
# no reprojection reward, no discriminator at all. See
# `src/models/supervised.py`'s module docstring for why that separation is
# the point: (D) can only say whether PPO recovered the signal, (E) asks
# whether the signal is there to recover in the first place.
#
# Three seeds (42/43/44), matching the standard the rest of this report holds
# itself to -- report.md has already retracted one single-seed claim
# ((B3)'s apparent seed-42 advantage), so a 1-seed (E) number would not meet
# the bar the rest of the document sets.
#
# Runs sequentially, not fanned out across cores like (C)/(D)'s CPU sweeps:
# there is one GPU here, not 32 cores, and training itself is already cheap
# (SGD over pre-loaded tensors, no env stepping, no forward-kinematics during
# training) -- see report.md once (E)'s numbers land for how long a seed
# actually took on this machine. Eval runs on CPU, matching every other
# --dump_scores pass in this project.
#
# Every step skips work whose output already exists, so a run stopped early
# (Ctrl+C, heat, time) resumes on the next invocation without redoing
# finished seeds. `src/train_supervised.py` also checkpoints periodically and
# on Ctrl+C within a single seed (--checkpoint_interval), but that
# in-progress file is not `regressor_final.pt` -- this script only treats a
# seed as done once that final file exists, so an interrupted seed reruns
# from scratch rather than resuming mid-epoch.
#
# NOTE on --lifted_h5: report.md's own reproduce blocks write the bare
# filename `lifted_movi_part1_upd2.h5`, which does not exist relative to the
# repo root on this checkout -- only `data/lifted_movi_part1_upd2.h5` does.
# That looks like a stale path in the docs (the file was likely moved into
# `data/` after those blocks were written), not a second, different file --
# same landmine `src.rewards.load_lifted_stats`'s docstring already warns
# about for the normalization jsons. This script uses the path that actually
# exists.
set -u

PY="${PY:-/c/Users/gusta/miniconda3/envs/rl_project/python.exe}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

H5=data/processed_movi.h5
NORM=data/normalization.json
LIFTED_H5=data/lifted_movi_part1_upd2.h5
OUT=checkpoints/exp_e
LOG=logs/exp_e
SCORES=eval_scores
mkdir -p "$OUT" "$LOG" "$SCORES"

echo "=== experiment (E) supervised regression, training start $(date) ==="
for seed in 42 43 44; do
  tag="s${seed}"
  if [ -f "$OUT/$tag/regressor_final.pt" ]; then
    echo "[skip] $tag already trained"
    continue
  fi
  echo "[train] $tag"
  "$PY" -u -m src.train_supervised \
      --h5_path "$H5" --norm_stats_path "$NORM" \
      --out_dir "$OUT/$tag" --seed "$seed" \
      2>&1 | tee "$LOG/train_${tag}.log"
done
echo "=== experiment (E) training done $(date) ==="

echo "=== experiment (E) eval start $(date) ==="
for seed in 42 43 44; do
  tag="s${seed}"
  ck="$OUT/$tag/regressor_final.pt"
  dump="$SCORES/supervised_${tag}.json"
  if [ ! -f "$ck" ]; then
    echo "!! MISSING $ck, skipping eval for $tag"
    continue
  fi
  if [ -f "$dump" ]; then
    echo "[skip] $dump already exists"
    continue
  fi
  echo "[eval] $tag"
  "$PY" -u -m src.evaluate \
      --processed_h5 "$H5" --norm_stats_path "$NORM" \
      --lifted_h5 "$LIFTED_H5" --joint_set j14 --split test --device cpu \
      --supervised_checkpoint "$ck" \
      --dump_scores "$dump" \
      2>&1 | tee "$LOG/eval_${tag}.log"
done
echo "=== experiment (E) eval done $(date) ==="

missing=0
for seed in 42 43 44; do
  d="$SCORES/supervised_s${seed}.json"
  if [ ! -f "$d" ]; then
    echo "!! MISSING $d"
    missing=1
  fi
done
[ "$missing" -eq 0 ] && echo "All 3 seeds present."

echo
echo "Aggregate into the report.md table with:"
echo "  python scripts/sweep_stats.py --table pampjpe --scores_dir $SCORES"
