#!/bin/bash
# Lighter re-run of scripts/run_eval_c.sh, after that script took the box down.
#
# Why the original crashed: it launched all 6 PA-MPJPE evals at once with only
# OMP_NUM_THREADS as a limiter. torch/BLAS did not honour it -- load average hit
# 425 on 32 cores while RAM stayed at 14/62 GB, so this was CPU oversubscription,
# not OOM. Env vars are advisory; taskset is not, so every process here is pinned
# to a fixed core set and at most 2 run at a time. Hard ceiling: 12 of 32 cores.
#
# Also: the original passed --lifted_h5 to all 6 evals, so each recomputed the
# same deterministic 187-clip baseline. It is computed once here and merged into
# the dumps afterwards, which keeps the JSON schema byte-compatible with the (B)
# dumps that sweep_stats.py reads -- 1122 redundant clip-evals become 187.
#
# Everything stays on --device cpu with the same joint set, split, clips and
# seeds as the (B) runs, so (C) remains comparable. Re-runnable: any step whose
# output already exists is skipped, so a crash costs only the work in flight.
set -u
PY=/home/annkle/miniconda/envs/smplerx/bin/python
ROOT=/home/annkle/rl-project
cd "$ROOT" || exit 1

export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6

LOG=logs/eval_c_light
mkdir -p eval_scores "$LOG"
BASE=eval_scores/_baseline_c.json

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
echo "=== (C) PA-MPJPE dumps, start $(date +%T) ==="
slot=0
for cond in feet_out feet_in; do
  for seed in 42 43 44; do
    out="eval_scores/gail_${cond}_s${seed}.json"
    if [ -s "$out" ]; then echo "[skip] $out"; continue; fi
    ck="checkpoints/gail_c/${cond}_s${seed}/actor_final.pt"
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
echo "=== (C) PA-MPJPE dumps, done $(date +%T) ==="

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
for p in sorted(pathlib.Path("eval_scores").glob("gail_feet_*_s4*.json")):
    d = json.load(open(p))
    if all(k in d for k in keys):
        print(f"[skip] {p.name} already carries the baseline")
        continue
    d.update({k: base[k] for k in keys})
    json.dump(d, open(p, "w"))
    print(f"[merged] {p.name}")
PYEOF

# ── Step 4: held-out reprojection, same clips as the (B) row ─────────────────
HO=eval_scores/heldout_test_gail_c.json
if [ -s "$HO" ]; then
  echo "[skip] held-out already at $HO"
else
  echo "=== (C) held-out reprojection, start $(date +%T) ==="
  run 0-5 "$LOG/heldout.log" \
    $PY -u scripts/heldout_eval.py \
      --sweep_dir checkpoints/gail_c \
      --runs feet_out_s42 feet_out_s43 feet_out_s44 \
             feet_in_s42 feet_in_s43 feet_in_s44 \
      --split test --n_clips 400 --n_frames 12 --clip_seed 42 \
      --device cpu --dump "$HO"
  echo "=== (C) held-out reprojection, done $(date +%T) ==="
fi

echo "=== aggregate ==="
$PY scripts/sweep_stats.py --table pampjpe --scores_dir eval_scores
echo "EVAL C COMPLETE $(date +%T)"
