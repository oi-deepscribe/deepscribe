#!/bin/bash

#SBATCH -c=30

luigi --module deepscribe.luigi.training TrainGBModelTask --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder data/processed/pfa_new \
      --modelsfolder models \
      --target-size 50 \
      --keep-categories '["na", "HAL", "iš", "MEŠ", "ma", "1", "du", "da"]'  \
      --fractions '[0.7, 0.1, 0.2]' \
      --model-definition data/model_defs/gb.json
