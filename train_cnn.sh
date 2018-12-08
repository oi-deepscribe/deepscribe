#!/bin/bash

python -m deepscribe.scripts.train_cnn --npz data/processed/PFA_Large/over_300.npz \
                                      --tensorboard logs/cnn \
                                      --split 0.9 \
                                      --bsize 128 \
                                      --epochs 100 \
                                      --output output/cnn
