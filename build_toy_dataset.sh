#/bin/bash

csvfolder=data/ochre/PFA_Hotspot_Cutouts

# collect CSV folders
csvfolders=""
for i in `seq 1 11`
do
  csvfolders+="$csvfolder/000image_$i.csv "
done

python -m deepscribe.scripts.select_classes --datafiles $csvfolders \
                                            --imgfolder data/ochre/PFA_Hotspot_Cutouts/Hotspot_images \
                                            --remove_prefix PFS \
                                            --classes 1 2 \
                                            --out data/processed/toy/original
