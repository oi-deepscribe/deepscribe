#!/bin/bash

#TODO: update this

luigi --module deepscribe.luigi.ml_input AssignDatasetTask --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder data/processed/pfa_new \
      --target-size 200 \
      --keep-categories '["1","2"]'  \
