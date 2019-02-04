#!/bin/bash

python -m deepscribe.scripts.train_cnn --npz data/processed/toy/1_2.npz \
                                      --tensorboard logs/cnn \
                                      --split 0.9 \
                                      --bsize 64 \
                                      --epochs 300 \
                                      --output output/cnn
