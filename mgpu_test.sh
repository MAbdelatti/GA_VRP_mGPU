#!/bin/bash

#SBATCH -t 00:30:00 # 30 minutes MAX
#SBATCH -p dgx

#nvidia-smi
python3 mgpu.py journal-set/X/X-n219-k73.vrp 20 0 20 60 30
