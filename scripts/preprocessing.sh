#!/bin/bash

#TODO: update this

luigi --module deepscribe.luigi.ml_input AssignDatasetTask --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder data/processed/pfa_new \
      --target-size 50 \
      --keep-categories '["1","2"]'  \
      --fractions '[0.7, 0.1, 0.2]' \
