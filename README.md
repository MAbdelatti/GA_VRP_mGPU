# GA_VRP_mGPU
An update and improvement of the GA for VRP on multiple GPUs
The vrp.sh file contains the following python command to run a program instance
```bash
python gpu.py <problem name> <# of geberations> <known optimal or 0> <population size multiplier> <crossover operator> <mutation operator>
```
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
