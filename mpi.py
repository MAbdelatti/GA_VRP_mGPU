# Example: mpi_script.py
from mpi4py import MPI
import gpuInfo

print(gpuInfo.get_gpu_info())

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"total of {size}")