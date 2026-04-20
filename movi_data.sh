#!/bin/bash
#SBATCH --job-name=movi2h5
#SBATCH --output=movi2h5_%j.log
#SBATCH --error=movi2h5_%j.err
#SBATCH --export=ALL,HYDRA_FULL_ERROR=1
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --nodelist=deepspeech
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00

echo "Job started on $SLURMD_NODENAME at $(date)"
echo "Using GPU(s): ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}"

CONDA_ROOT="/nfs/tts2/home/annkle/new_conda/miniconda3/"
CONDA_ENV_NAME="miss_stupid"

# "/nfs/tts2/home/annkle/miniconda3/"
# CONDA_ROOT= "/nfs/deepspeech/home/annkle/miniconda3/"
eval "$("$CONDA_ROOT/bin/conda" shell.bash hook)"

conda clean -a -y
pip3 cache purge 
# conda activate /nfs/tts2/home/annkle/miniconda3/envs/anya_env
conda activate "$CONDA_ENV_NAME"

rm -rf /home/annkle/lightning-hydra-template/body_models/models_lockedhead/smplx/__pycache__
# pip install h5py

python pack_movi_hdf5.py \
        --mat_dir    /nfs/deepspeech/home/annkle/mosh-processing/AMASS/ \
        --out_hdf5   movi.h5 \
        --train_frac 0.80 \
        --val_frac   0.10 \
        --seed       42

conda deactivate
