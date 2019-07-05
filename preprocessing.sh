#!/bin/bash

luigi --module deepscribe.luigi.preprocessing OchreToHD5Task --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder data/processed/pfa_new
