#!/bin/bash

#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eddiecwilliams@gmail.com
#SBATCH --output=confusion-10-%j.out
#SBATCH --error=confusion-10-%j.err
#SBATCH --mem=16G

module load cuda/9.1

SIGNS='["na","HAL","iš","MEŠ","ma","1","du","da","AN","AŠ"]'
#SIGNS='["na","HAL"]'
#SIGNS='["na","HAL","iš","MEŠ","ma","1","du","da","AN","AŠ"]'


luigi --module deepscribe.pipeline.analysis RunAnalysisOnTestDataTask --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder ../deepscribe-data/processed/pfa_new \
      --modelsfolder models \
      --target-size 50 \
      --keep-categories $SIGNS  \
      --fractions '[0.7, 0.1, 0.2]' \
      --model-definition data/model_defs/alexnet-small-earlystopping-10-reweighted.json \
      --k 2