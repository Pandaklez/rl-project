#!/bin/bash
# Evaluations for the (D) column of report.md, from checkpoints/exp_d.
#
# Same two metrics and the same discipline as run_eval_c_light.sh, for the same
# reason: the unpinned run_eval_c.sh launched six evals at once with only
# OMP_NUM_THREADS as a limiter, torch/BLAS did not honour it, and load average
# hit 425 on 32 cores while RAM stayed at 14/62 GB. That was CPU
# oversubscription, not OOM. Env vars are advisory; taskset is not, so every
# process here is pinned and at most 2 run at a time. Hard ceiling: 12 of 32.
#
#   1. PA-MPJPE on test, via src/evaluate.py --dump_scores. Named
#      d_<cond>_s<seed>.json because sweep_stats.py's D_VARIANTS looks for
#      exactly that in --scores_dir.
#   2. Held-out reprojection, via scripts/heldout_eval.py, with the same
#      --split test --n_clips 400 --clip_seed 42 as the (B) and (C) rows, so all
#      three describe identical clips. The script prints the lifted error per
#      run; it must read 12.1368 px for every one of them.
#
# The lifted baseline is computed once and merged into each dump rather than
# recomputed per checkpoint -- it does not depend on the checkpoint, so six
# identical 187-clip evals would be five wasted. The merge keeps the JSON schema
# byte-compatible with the (B)/(C) dumps sweep_stats.py reads.
#
# Re-runnable: any step whose output already exists is skipped, so a crash costs
# only the work in flight.
set -u
PY=/home/annkle/miniconda/envs/smplerx/bin/python
ROOT=/home/annkle/rl-project
cd "$ROOT" || exit 1

export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6

LOG=logs/eval_d
mkdir -p eval_scores "$LOG"
BASE=eval_scores/_baseline_c.json      # checkpoint-independent; shared with (C)

run () {  # run <core-range> <logfile> <cmd...>
  local cores=$1 log=$2; shift 2
  nice -n 10 taskset -c "$cores" "$@" > "$log" 2>&1
}

# ── Step 1: the lifted baseline, once, on 6 cores ────────────────────────────
if [ -s "$BASE" ]; then
  echo "[skip] baseline already at $BASE"
else
  echo "=== baseline, start $(date +%T) ==="
  run 0-5 "$LOG/baseline.log" \
    $PY -u -m src.evaluate \
      --processed_h5 data/processed_movi.h5 \
      --lifted_h5 lifted_movi_part1_upd2.h5 \
      --joint_set j14 --betas gt --split test --device cpu \
      --dump_scores "$BASE"
  echo "=== baseline, done $(date +%T) ==="
fi

# ── Step 2: the 6 model evals, 2 at a time, 6 cores each ─────────────────────
echo "=== (D) PA-MPJPE dumps, start $(date +%T) ==="
slot=0
for cond in mse1 mse10; do
  for seed in 42 43 44; do
    out="eval_scores/d_${cond}_s${seed}.json"
    if [ -s "$out" ]; then echo "[skip] $out"; continue; fi
    ck="checkpoints/exp_d/${cond}_s${seed}/actor_final.pt"
    if [ ! -f "$ck" ]; then echo "!! missing $ck"; continue; fi
    if [ "$slot" -eq 0 ]; then cores=0-5; else cores=6-11; fi
    echo "  -> ${cond}_s${seed} on cores $cores"
    run "$cores" "$LOG/pampjpe_${cond}_s${seed}.log" \
      $PY -u -m src.evaluate \
        --processed_h5 data/processed_movi.h5 \
        --joint_set j14 --betas gt --split test --device cpu \
        --checkpoint "$ck" --dump_scores "$out" &
    slot=$((1 - slot))
    [ "$slot" -eq 0 ] && wait   # both slots busy -> drain before refilling
  done
done
wait
echo "=== (D) PA-MPJPE dumps, done $(date +%T) ==="

# ── Step 3: merge the shared baseline into each dump ─────────────────────────
# sweep_stats.py takes the baseline from whichever dump it reads first and pairs
# per-clip scores against it, so every dump must carry those four fields.
$PY - "$BASE" <<'PYEOF'
import json, sys, pathlib
base = json.load(open(sys.argv[1]))
keys = ["pa_mpjpe_lifted_raw", "n_clips_baseline",
        "per_clip_lifted", "per_cam_lifted", "per_cam_keys_lifted"]
missing = [k for k in keys if k not in base]
if missing:
    sys.exit(f"baseline dump is missing {missing}")
for p in sorted(pathlib.Path("eval_scores").glob("d_mse*_s4*.json")):
    d = json.load(open(p))
    if all(k in d for k in keys):
        print(f"[skip] {p.name} already carries the baseline")
        continue
    d.update({k: base[k] for k in keys})
    json.dump(d, open(p, "w"))
    print(f"[merged] {p.name}")
PYEOF

# ── Step 4: held-out reprojection, same clips as the (B)/(C) rows ────────────
HO=eval_scores/heldout_test_d.json
if [ -s "$HO" ]; then
  echo "[skip] held-out already at $HO"
else
  echo "=== (D) held-out reprojection, start $(date +%T) ==="
  run 0-5 "$LOG/heldout.log" \
    $PY -u scripts/heldout_eval.py \
      --sweep_dir checkpoints/exp_d \
      --runs mse1_s42 mse1_s43 mse1_s44 \
             mse10_s42 mse10_s43 mse10_s44 \
      --split test --n_clips 400 --n_frames 12 --clip_seed 42 \
      --device cpu --dump "$HO"
  echo "=== (D) held-out reprojection, done $(date +%T) ==="
fi

echo "=== aggregate ==="
$PY scripts/sweep_stats.py --table pampjpe --scores_dir eval_scores
echo "EVAL D COMPLETE $(date +%T)"
