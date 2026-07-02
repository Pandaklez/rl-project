#!/usr/bin/env bash
# Run SMPLer-X inference on the remaining PG2 subjects: 8, 9, 73-90
# Must be run from inside smpler-x-main/
set -uo pipefail

cd "$(dirname "$0")" || exit 1

CONDA_ENV=smplerx
CKPT_NAME=smpler_x_b32
NUM_GPUS=1
FORMAT=avi
FPS=30
VIDEO_DIR=/home/annkle/rl-project/demo/videos/PG2_avi

REMAINING_SUBJECTS=(8 9 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90)

for ID in "${REMAINING_SUBJECTS[@]}"; do
    VIDEO_STEM="F_PG2_Subject_${ID}_L"
    VIDEO_FILE="${VIDEO_DIR}/${VIDEO_STEM}.${FORMAT}"
    IMG_PATH="../demo/images/${VIDEO_STEM}"
    SAVE_DIR="../demo/results/${VIDEO_STEM}"
    SMPLX_DIR="${SAVE_DIR}/smplx"

    if [[ -d "${SMPLX_DIR}" ]]; then
        echo "[SKIP] ${VIDEO_STEM} (smplx dir exists)"
        continue
    fi

    if [[ ! -f "${VIDEO_FILE}" ]]; then
        echo "[SKIP] ${VIDEO_STEM} (video file not found: ${VIDEO_FILE})"
        continue
    fi

    echo "=== Processing ${VIDEO_STEM} ==="

    mkdir -p "${IMG_PATH}"
    mkdir -p "${SAVE_DIR}"

    existing=$(find "${IMG_PATH}" -name "*.jpg" -type f | wc -l)
    if [[ $existing -eq 0 ]]; then
        ffmpeg -i "${VIDEO_FILE}" -f image2 -vf fps=${FPS}/1 -q:v 2 "${IMG_PATH}/%06d.jpg"
    fi

    frame_count=$(find "${IMG_PATH}" -name "*.jpg" -type f | wc -l)
    echo "  frames: ${frame_count}"

    conda run -n ${CONDA_ENV} python inference.py \
        --num_gpus ${NUM_GPUS} \
        --exp_name "output/demo_inference_${VIDEO_STEM}" \
        --pretrained_model ${CKPT_NAME} \
        --agora_benchmark agora_model \
        --img_path "${IMG_PATH}" \
        --start 1 \
        --end "${frame_count}" \
        --output_folder "${SAVE_DIR}" \
        --no_render

    echo "  done -> ${SAVE_DIR}/smplx"

    # Remove extracted frames to save disk space
    rm -rf "${IMG_PATH}"
done

echo "All remaining PG2 subjects processed."
