#!/bin/bash

python -m deepscribe.scripts.train_cnn --npz data/processed/PFA_Large/over_100.npz \
                                      --tensorboard logs/cnn \
                                      --split 0.9 \
                                      --bsize 2048 \
                                      --epochs 10 \
                                      --output output/cnn
