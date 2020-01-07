#!/bin/bash

#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eddiecwilliams@gmail.com
#SBATCH --output=confusion-4-%j.out
#SBATCH --error=confusion-4-%j.err
#SBATCH --mem=16G

module load cuda/9.1

SIGNS='["na","HAL","iš","MEŠ"]'
#SIGNS='["na","HAL"]'
#SIGNS='["na","HAL","iš","MEŠ","ma","1","du","da","AN","AŠ"]'


luigi --module deepscribe.luigi.model_selection PlotConfusionMatrixTask --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder ../deepscribe-data/data/processed/pfa_new \
      --modelsfolder models \
      --target-size 50 \
      --keep-categories $SIGNS  \
      --fractions '[0.7, 0.1, 0.2]' \
      --model-definition data/model_defs/alexnet-small.json