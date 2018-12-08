#!/bin/bash

python -m deepscribe.scripts.train_mlp --npz data/processed/PFA_Large/over_100.npz \
                                      --tensorboard logs/mlp \
                                      --split 0.9 \
                                      --nlayers 5 \
                                      --lsize 512 \
                                      --bsize 128 \
                                      --epochs 50 \
                                      --output output/mlp
