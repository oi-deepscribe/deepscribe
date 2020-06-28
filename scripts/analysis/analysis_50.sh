#!/bin/bash

#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 #requesting 1 CPU
#SBATCH --gres=gpu:1     # Request 1 GPU
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eddiecwilliams@gmail.com
#SBATCH --output=slogs/confusion-50-%j.out
#SBATCH --error=slogs/confusion-50-%j.err
#SBATCH --mem=16G

# module load cuda/9.1


PYTHONPATH="." luigi --module deepscribe.pipeline.analysis TrainAndAnalyze --local-scheduler \
      --imgarchive "/local/ecw/deepscribe-data/pfa/a_pfa_cleaned.h5" \
      --target-size 50 \
      --keep-categories "/local/ecw/deepscribe/notebooks/top50.txt" \
      --fractions '[0.7, 0.1, 0.2]' \
      --sigma 0.0 \
      --modelsfolder models_cleaned \
      --lr 0.001 \
      --optimizer adam \
      --epochs 128 \
      --early-stopping 10 \
      --reduce-lr 5 \
      --shear 00.0 \
      --zoom 0.2 \
      --width-shift 0.0 \
      --height-shift 0.0 \
      --rotation-range 00.0 \
      --bsize 32 \
      --histogram "adaptive" \
      # --shadow-val-range "[-2.0, 2.0]" \
      # --num-shadows 1 
