#!/bin/sh
#BSUB -J uc_aisles
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -q milan

#BSUB -R "rusage[mem=32GB]"

### ------------- specify wall-clock time (max allowed is 12:00)---------------- 
#BSUB -W 12:00

#BSUB -o outputs/%J.out
#BSUB -e outputs/%J.err

source .venv/bin/activate
#python pipeline.py
python experiments/cf_user_clusters_aisles_experiments.py \
  --n_features "500,1000" \
  --reg_vals "0.0001,0.001,0.01" \
  --damping_vals "5,10" \
  --bias 1 \
  --n_clusters 50,100,150,200