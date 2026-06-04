#!/usr/bin/env bash
# Batch SMPLer-X inference over PG1 and PG2 videos.
# Run from the repo root:  bash smpler-x-main/batch_inference.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_ENV=smplerx
CKPT_NAME=smpler_x_b32
FPS=30
NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0

VIDEO_DIRS=(
    "$REPO_ROOT/demo/videos/PG1_avi"
    "$REPO_ROOT/demo/videos/PG2_avi"
)

cd "$SCRIPT_DIR"

for VIDEO_DIR in "${VIDEO_DIRS[@]}"; do
    for VIDEO_PATH in "$VIDEO_DIR"/*.avi; do
        VIDEO_NAME="$(basename "$VIDEO_PATH" .avi)"
        IMG_PATH="$REPO_ROOT/demo/images/$VIDEO_NAME"
        SAVE_DIR="$REPO_ROOT/demo/results/$VIDEO_NAME"
        SMPLX_DIR="$SAVE_DIR/smplx"

        # skip if already fully processed
        if [[ -d "$SMPLX_DIR" && "$(find "$SMPLX_DIR" -name '*.npz' | wc -l)" -gt 0 ]]; then
            echo "[SKIP] $VIDEO_NAME (smplx dir exists)"
            continue
        fi

        echo "=== Processing $VIDEO_NAME ==="

        # extract frames (skip if already done)
        mkdir -p "$IMG_PATH"
        existing=$(find "$IMG_PATH" -name '*.jpg' | wc -l)
        if [[ $existing -eq 0 ]]; then
            ffmpeg -i "$VIDEO_PATH" -f image2 -vf fps=${FPS}/1 -q:v 2 \
                "$IMG_PATH/%06d.jpg" -loglevel error
        fi

        end_count=$(find "$IMG_PATH" -name '*.jpg' | wc -l)
        echo "  frames: $end_count"

        mkdir -p "$SAVE_DIR"
        if conda run -n "$CONDA_ENV" python inference.py \
            --num_gpus    "$NUM_GPUS" \
            --exp_name    "output/demo_${VIDEO_NAME}" \
            --pretrained_model "$CKPT_NAME" \
            --agora_benchmark agora_model \
            --img_path    "$IMG_PATH" \
            --start       1 \
            --end         "$end_count" \
            --output_folder "$SAVE_DIR" \
            --no_render; then
            echo "  done -> $SMPLX_DIR"
        else
            echo "  [ERROR] inference failed for $VIDEO_NAME, skipping"
            rm -rf "$SAVE_DIR"
        fi

        # free disk space — frames are no longer needed once results are saved
        rm -rf "$IMG_PATH"
    done
done

echo ""
echo "=== All done ==="
