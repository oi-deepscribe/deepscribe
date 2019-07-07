#!/bin/bash

luigi --module deepscribe.luigi.image_processing StandardizeImageSize --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder data/processed/pfa_new \
      --target-size 300
