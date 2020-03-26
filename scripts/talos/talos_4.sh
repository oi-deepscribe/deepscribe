#!/bin/bash

#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eddiecwilliams@gmail.com
#SBATCH --output=top-50-other-talos-%j.out
#SBATCH --error=top-50-other-talos-%j.err
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
      --model-definition data/talos_params/short_test.json
