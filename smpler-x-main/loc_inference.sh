#!/usr/bin/env bash
set -euxo pipefail

# run from inside the package directory so top-level imports work
cd "$(dirname "$0")" || exit 1

INPUT_VIDEO=${1:-S2_PG1_Subject_66_L}
FORMAT=${2:-avi}
FPS=${3:-30}
CKPT_NAME=smpler_x_b32  # Model name (without .pth.tar)
CONDA_ENV=smplerx

NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0
JOB_NAME=inference_${INPUT_VIDEO}

IMG_PATH=../demo/images/${INPUT_VIDEO}
SAVE_DIR=../demo/results/${INPUT_VIDEO}

# Verify GPU availability
echo "Checking GPU availability..."
conda run -n ${CONDA_ENV} python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}'); print(f'Using GPU: {torch.cuda.is_available()}')"

# video to images (skip if already extracted)
mkdir -p $IMG_PATH
mkdir -p $SAVE_DIR

existing=$(find "$IMG_PATH" -name "*.jpg" -type f | wc -l)
if [[ $existing -eq 0 ]]; then
    ffmpeg -i /home/annkle/rl-project/demo/videos/${INPUT_VIDEO}.${FORMAT} \
        -f image2 -vf fps=${FPS}/1 -q:v 2 \
        ../demo/images/${INPUT_VIDEO}/%06d.jpg
fi

end_count=$(find "$IMG_PATH" -name "*.jpg" -type f | wc -l)
echo "Total frames: $end_count"

# inference
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

conda run -n ${CONDA_ENV} python inference.py \
    --num_gpus ${NUM_GPUS} \
    --exp_name output/demo_${JOB_NAME} \
    --pretrained_model ${CKPT_NAME} \
    --agora_benchmark agora_model \
    --img_path ${IMG_PATH} \
    --start 1 \
    --end $end_count \
    --output_folder ${SAVE_DIR} \
    --no_render

echo "Done. SMPL-X params saved to ${SAVE_DIR}/smplx/"
# Usage: bash smpler-x-main/loc_inference.sh [VIDEO_NAME] [FORMAT] [FPS]