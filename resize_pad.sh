#!/bin/bash

python -m deepscribe.scripts.resize_and_pad --infolder data/processed/toy/original/1 \
                                            --target_size 100 \
                                            --outfolder data/processed/toy/padded/1

python -m deepscribe.scripts.resize_and_pad --infolder data/processed/toy/original/2 \
                                            --target_size 100 \
                                            --outfolder data/processed/toy/padded/2
