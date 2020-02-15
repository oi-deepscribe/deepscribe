#!/bin/bash

luigi --module deepscribe.pipeline.selection SelectDatasetTask --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder ../deepscribe-data/processed/pfa_new \
      --target-size 50 \
      --keep-categories '["na", "HAL"]'  \
      --fractions '[0.7, 0.1, 0.2]' \
      --num-augment 1 \
