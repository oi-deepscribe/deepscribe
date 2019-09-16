#!/bin/bash

#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU

module load cuda/9.1


luigi --module deepscribe.luigi.training RunTalosScanTask --local-scheduler \
      --imgfolder data/ochre/a_pfa \
      --hdffolder data/processed/pfa_new \
      --target-size 50 \
      --keep-categories '["1","2"]'  \
      --fractions '[0.7, 0.1, 0.2]' \
      --talos-params data/talos_params/simple_cnn.json \
      --subsample 1

