#!/bin/bash


PYTHONPATH="." luigi --module deepscribe.pipeline.selection SelectDatasetTask --local-scheduler \
      --imgarchive "/local/ecw/deepscribe-data/pfa/a_pfa_cleaned.h5" \
      --target-size 50 \
      --keep-categories /local/ecw/deepscribe/data/charsets/top50.txt \
      --fractions '[0.7, 0.1, 0.2]' \
      --sigma 0.0