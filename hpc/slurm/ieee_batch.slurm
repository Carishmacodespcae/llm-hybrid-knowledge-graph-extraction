#!/bin/bash
#SBATCH --job-name=ieee_chunk
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:4
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --time=40:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/prod_%A_%a.out
#SBATCH --error=logs/prod_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL

source ~/.bashrc
conda activate llm-env

cd /mnt/vast-kisski/home/c.ahamedkhan/u25954/llm_bio_extraction

export MY_SCRATCH=/mnt/vast-kisski/home/c.ahamedkhan/u25954/tmp
export TMPDIR=$MY_SCRATCH
export VLLM_WORKER_MULTIPROC_TMPDIR=$MY_SCRATCH
export TRITON_CACHE_DIR=$MY_SCRATCH/triton_cache
export TORCHINDUCTOR_CACHE_DIR=$MY_SCRATCH/torchinductor_$USER
export VLLM_CACHE_ROOT=$MY_SCRATCH/vllm_cache
mkdir -p $TMPDIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $VLLM_CACHE_ROOT

echo "=== Node: $(hostname) ==="
echo "=== Array Job: $SLURM_ARRAY_JOB_ID | Task: $SLURM_ARRAY_TASK_ID ==="
nvidia-smi --list-gpus

# ── Chunk ranges for 390,928 records split into 4 equal parts ─────────────────
case $SLURM_ARRAY_TASK_ID in
  0) START=1;      END=97732  ;;
  1) START=97733;  END=195464 ;;
  2) START=195465; END=293196 ;;
  3) START=293197; END=390928 ;;
esac

echo "Processing lines $START – $END"
python ieee_prod_parallel.py --start $START --end $END