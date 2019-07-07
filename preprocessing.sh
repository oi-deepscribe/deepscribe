#!/bin/bash

luigi --module deepscribe.luigi.ml_input AssignDatasetTask --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder data/processed/pfa_new \
      --target-size 300 \
      --keep-categories '["1","2"]'  \
