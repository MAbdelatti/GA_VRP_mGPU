# GA_VRP_mGPU
An update and improvement of the GA for VRP on multiple GPUs

example:

```bash
python gpu.py X-n1001-k43 50000 0 10 60 30
```
command to run an instance on a slurm workload manager:
```bash
sbatch vrp.sh
tail -f <the returned file from the above command>
```
 or use the python comman directly if working on a local machine
