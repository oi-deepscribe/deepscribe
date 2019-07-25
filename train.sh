#!/bin/bash

luigi --module deepscribe.luigi.training TrainModelTask --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder data/processed/pfa_new \
      --target-size 200 \
      --epochs 5 \
      --keep-categories '["1","2"]'  \
