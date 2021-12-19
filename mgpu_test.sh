#!/bin/bash

#SBATCH -t 02:00:00 # 02 hours MAX
#SBATCH -p dgx
#SBATCH -e slurm-%j.err

#nvidia-smi
python3 mgpu.py journal-set/X/X-n856-k95.vrp 10000 0 20 60 30
