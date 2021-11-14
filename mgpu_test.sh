#!/bin/bash

#SBATCH -t 00:30:00 # 30 minutes MAX
#SBATCH -p dgx

#nvidia-smi
python3 mgpu_test.py
