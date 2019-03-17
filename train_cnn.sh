#!/bin/bash


pythonw -m deepscribe.scripts.training.train_cnn --npz data/processed/toy/1_2_3/1_2_3.npz \
                                      --tensorboard logs/cnn \
                                      --output output/cnn
