#!/bin/bash

python -m deepscribe.scripts.build_dataset --dataset data/processed/toy/padded \
                                          --npz data/processed/toy/1_2.npz
