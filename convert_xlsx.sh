#!/bin/bash

#converts XLSX to csv

infolder=data/ochre/PFA_Hotspot_Cutouts
outfolder=$infolder

for i in $infolder/*.xlsx; do
  base=$(basename -- "$i")
  filename="${base%.*}"
  xlsx2csv $i $outfolder/$filename.csv
done
