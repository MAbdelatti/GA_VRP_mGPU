# GA_VRP_mGPU
An update and improvement of the GA for VRP on multiple GPUs.

The following example runs the algorithm on a non-job scheduling platform utilizing **8 GPUs** on a benchmark problem named **Golden_12**, for **10,000** generations, **0** optimal value (to get the lowest possible value), population size of **20n**, crossover rate of **60%**, and a mutation rate of **30%**:

```bash
python3 mgpu.py journal-set/Golden/Golden_12.vrp 10000 0 20 60 30 >> Golden12.out &
```

To utilize different GPU arrangements (e.g., 1, 2, or 4) replace the **mgpu.py** file with either: **mgpu-1.py, mgpu-2.py, or mgpu-4.py**, respectively.

The following example runs the algorithm with the same setings dicussed above on platform with **slurm** workload manager:

```bash
srun -p dgx python3 -u mgpu.py journal-set/Golden/Golden_12.vrp 10000 0 20 60 30 >> Golden12.out &
```
