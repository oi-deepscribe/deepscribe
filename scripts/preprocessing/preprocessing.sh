#!/bin/bash

luigi --module deepscribe.luigi.ml_input AssignDatasetTask --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder data/processed/pfa_new \
      --target-size 50 \
      --keep-categories '["na", "HAL"]'  \
      --fractions '[0.7, 0.1, 0.2]' \
      --num-augment 10 \