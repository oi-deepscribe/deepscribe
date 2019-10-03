#!/bin/bash

#SBATCH -c=30

luigi --module deepscribe.luigi.training TrainRFModelTask --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder data/processed/pfa_new \
      --modelsfolder models \
      --target-size 50 \
      --keep-categories '["1","2", "na"]'  \
      --fractions '[0.7, 0.1, 0.2]' \
      --model-definition data/model_defs/rf.json
