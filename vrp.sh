#!/bin/bash

#SBATCH -t 00:30:00 # 30 minutes MAX
#SBATCH -p dgx

#nvidia-smi
python3 gpu.py P-n16-k8 2000 450 20 60 30
