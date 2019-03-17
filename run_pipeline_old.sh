#!/bin/bash

csvfolder=data/ochre/PFA_Hotspot_Cutouts

#min examples required
NEXAMPLES=300

# collect CSV folders
csvfolders=""
for i in `seq 1 11`
do
  csvfolders+="$csvfolder/000image_$i.csv "
done

python -m deepscribe.scripts.pipeline --datafiles $csvfolders \
                      --remove_prefix PFS \
                      --imgfolder data/ochre/PFA_Hotspot_Cutouts/Hotspot_images \
                      --examples_req $NEXAMPLES \
                      --min_size 25 25 \
                      --blur_thresh 75 \
                      --resize 50 50 \
                      --outfile data/processed/PFA_Large/over_$NEXAMPLES.npz
