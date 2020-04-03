#!/bin/bash


SIGNS='["na","HAL","iš","MEŠ"]'
#SIGNS='["na","HAL","iš","MEŠ","ma","1","du","da","AN","AŠ"]'

luigi --module deepscribe.pipeline.selection SelectDatasetTask --local-scheduler \
      --imgfolder ../deepscribe-data/ochre/a_pfa \
      --hdffolder ../deepscribe-data/processed/pfa_new \
      --target-size 50 \
      --keep-categories $SIGNS \
      --fractions '[0.7, 0.1, 0.2]' \
      --sigma 0.5