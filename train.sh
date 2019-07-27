#!/bin/bash

luigi --module deepscribe.luigi.training TrainModelTask --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder data/processed/pfa_new \
      --target-size 200 \
      --epochs 5 \
      --batch_size 32 \
      --classes '["1","2"]'  \
      --data_split '[0.7, 0.1, 0.2]'
