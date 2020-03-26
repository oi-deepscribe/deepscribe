#!/bin/bash

#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eddiecwilliams@gmail.com
#SBATCH --output=4-talos-%j.out
#SBATCH --error=4-talos-%j.err
#SBATCH --mem=16G

module load cuda/9.1

SIGNS='["na","HAL","iš","MEŠ"]'

luigi --module deepscribe.pipeline.training RunTalosScanTask --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder ../deepscribe-data/processed/pfa_new \
      --modelsfolder models \
      --target-size 50 \
      --keep-categories $SIGNS  \
      --fractions '[0.7, 0.1, 0.2]' \
      --subsample 1 \
      --model-definition data/talos_params/varied_knums.json
