#!/bin/sh
#BSUB -J test_ct_aks 
#BSUB -n 1
#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=256GB]"

### ------------- specify wall-clock time (max allowed is 12:00)---------------- 
#BSUB -W 06:00

#BSUB -o outputs/%J.out
#BSUB -e outputs/%J.err

source .venv/bin/activate
python pipeline.py
