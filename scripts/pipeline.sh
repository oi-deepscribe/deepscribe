#!/bin/bash

# new image processing pipeline

csvfolder=data/ochre/PFA_Hotspot_Cutouts

# collect CSV folders
csvfolders=""
for i in `seq 1 11`
do
  csvfolders+="$csvfolder/000image_$i.csv "
done


classes="1 2 3 10"
#TODO: merge this
prefix="1_2_3_10"


originalimg=data/processed/toy/$prefix/original

mkdir -p $originalimg

# select classes from dataset
python -m deepscribe.scripts.select_classes --datafiles $csvfolders \
                                            --imgfolder data/ochre/PFA_Hotspot_Cutouts/Hotspot_images \
                                            --remove_prefix PFS \
                                            --classes $classes \
                                            --out $originalimg

# pad and process images
processimg=data/processed/toy/$prefix/padded

mkdir -p $processimg

for i in $classes; do
  python -m deepscribe.scripts.resize_and_pad --infolder $originalimg/$i \
                                              --target_size 100 \
                                              --outfolder $processimg/$i
done

# combine into one .npy file

outnpz=data/processed/toy/$prefix/$prefix.npz

python -m deepscribe.scripts.build_dataset --dataset $processimg \
                                          --npz $outnpz
