#!/bin/sh
#BSUB -J __256gb_memory
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -q milan

#BSUB -R "rusage[mem=256GB]"

### ------------- specify wall-clock time (max allowed is 12:00)---------------- 
#BSUB -W 06:00

#BSUB -o outputs/%J.out
#BSUB -e outputs/%J.err

source .venv/bin/activate
#python pipeline.py
python collaborative_filter_experiments.py \
  --n_features "500,750,1000,1250,1500" \
  --reg_vals "0.001,0.01,0.1" \
  --bias 0