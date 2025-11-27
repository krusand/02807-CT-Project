#!/bin/sh
#BSUB -J __256gb_memory
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -q milan

#BSUB -R "rusage[mem=64GB]"

### ------------- specify wall-clock time (max allowed is 12:00)---------------- 
#BSUB -W 12:00

#BSUB -o outputs/%J.out
#BSUB -e outputs/%J.err

source .venv/bin/activate
#python pipeline.py
python experiments/mba_experiments.py \
  --support_grids "0.00001,0.0001" \
  --min_confidences "0.001,0.005,0.01,0.05,0.1 \