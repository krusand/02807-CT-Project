#!/bin/sh
#BSUB -J mba
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -q milan

#BSUB -R "rusage[mem=64GB]"

### ------------- specify wall-clock time (max allowed is 12:00)---------------- 
#BSUB -W 12:00

#BSUB -o outputs/%J.out
#BSUB -e outputs/%J.err

source .venv/bin/activate

python pipeline.py

