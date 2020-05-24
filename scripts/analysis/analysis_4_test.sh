#!/bin/bash

#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3 #requesting 3 CPUs
#SBATCH --gres=gpu:1     # Request 1 GPU
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eddiecwilliams@gmail.com
#SBATCH --output=slogs/confusion-4-%j.out
#SBATCH --error=slogs/confusion-4-%j.err
#SBATCH --mem=16G

module load cuda/9.1

luigi --module deepscribe.pipeline.analysis TrainAndAnalyze --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder ../deepscribe-data/processed/pfa_new \
      --modelsfolder models \
      --target-size 50 \
      --keep-categories data/charsets/top4.txt  \
      --fractions '[0.7, 0.1, 0.2]' \
      --epochs 2 \
      --bsize 32