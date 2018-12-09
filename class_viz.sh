#!/bin/bash

python -m deepscribe.scripts.visualize_classes --npz data/processed/PFA_Large/thresh_over_300.npz \
                                                --nsamples 30 \
                                                --out output/dataviz/thresh/
