#!/bin/bash

data=data/processed/over_30

python src/cnn.py --train_images $data/train --train_labels $data/train.csv \
                  --test_images $data/test --test_labels $data/test.csv \
                  --flatten \
                  --output output
