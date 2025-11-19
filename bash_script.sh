#!/bin/sh
### ------------- specify job name ----------------
#BSUB -J test_ct_aks 
### ------------- specify number of cores ----------------
#BSUB -n 1
#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=20GB]"

### ------------- specify wall-clock time (max allowed is 12:00)---------------- 
#BSUB -W 01:00

#BSUB -o outputs/%J/script.out
#BSUB -e outputs/%J/script.err

source .venv/bin/activate
python pipeline.py
