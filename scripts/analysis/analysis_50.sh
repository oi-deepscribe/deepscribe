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
      --imgfolder data/ochre/a_pfa \
      --hdffolder ../deepscribe-data/processed/pfa_new \
      --modelsfolder models \
      --target-size 50 \
      --keep-categories data/charsets/top50.txt \
      --lr 0.001 \
      --fractions '[0.7, 0.1, 0.2]' \
      --epochs 54 \
      --shear 15.0 \
      --zoom 0.0 \
      --width-shift 0.2 \
      --height-shift 0.2 \
      --l2 0.0001 \
      --bsize 32
