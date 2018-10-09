#!/bin/bash

python src/pipeline.py --datacsv data/ochre/imageIndex.csv \
                      --remove_prefix PFS \
                      --imgfolder data/ochre/images_PFA \
                      --examples_req 30 \
                      --split 0.9 \
                      --resize 100 100 \
                      --outfolder data/processed/over_30
