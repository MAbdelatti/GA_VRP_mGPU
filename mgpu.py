#!/usr/bin/env python
# -------- Start of the importing part -----------
import gpu
import readInput
import gpuInfo
from numba import cuda
from numba.cuda import driver
import concurrent.futures
from timeit import default_timer as timer
import numpy as np
import sys
import gpuGrid
import val
import time
from mpi4py import MPI

# use maximum length in screen output
np.set_printoptions(threshold=sys.maxsize)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Retrieve GPU topology info:
gpu_count, nodeList, nodeSize, gpu_types, gpu_topology = gpuInfo.get_gpu_info()
gpuInfo.transfer_gpu_info(rank, gpu_count, gpu_types, gpu_topology, nodeSize, nodeList)
# gpuInfo.print_allocation_info(all_allocated_gpu_count, nodeList, all_gpu_types, all_gpu_topology, nodeSize)
# read problem file and save vrp_capacity, data, opt into basic_arguments list
if rank == 0: 
    print("Reading data file...", end=" ")

basic_arguments = readInput.readInput()
basic_arguments.append(sys.argv[1])  # filename

val = val.VRP(sys.argv[1], basic_arguments[1].shape[0])
n = int(sys.argv[4])
node_count = basic_arguments[1].shape[0]
totalpopsize = -(-(n * (node_count - 1)) // 1000) * 1000

exit()

popsize = min(totalpopsize // gpu_count, int(50e3))
print(
    "Population size assigned to be {}*n with {} per GPU. Total of {}.  \n".format(
        n, popsize, totalpopsize
    )
)

"""Basic arguments to be passed to the kernel:
vrp_capacity, data, known optimal cost, filename, gpu_count, population size, crossover_prob,
mutation_prob, popsize/gpu, crossover_points, blocks, threads_per_block, generations, r_flag"""
basic_arguments.append(gpu_count)  # gpu_count
basic_arguments.append(n)  # n
basic_arguments.append(int(sys.argv[5]))  # crossover_prob
basic_arguments.append(int(sys.argv[6]))  # mutation_prob
basic_arguments.append(popsize)  # popsize
basic_arguments.append(1)  # crossover_points


# GPU grid configurations:
grid = gpuGrid.GRID()
blocks_x, blocks_y = grid.blockAlloc(node_count, float(n))
tpb_x, tpb_y = grid.threads_x, grid.threads_y
basic_arguments.append((blocks_x, blocks_y))  # blocks
basic_arguments.append((tpb_x, tpb_y))  # threads_per_block

print(grid, "\n")

try:
    generations = int(sys.argv[2])
except:
    print("No generation limit provided, taking 2000 generations...")
    generations = 2000

basic_arguments.append(generations)

r_flag = 99999  # A flag for removal/replacement
basic_arguments.append(r_flag)

try:
    # Call function in single-thread-multi-GPU model
    # for GPU_ID in range(gpu_count):
    #         gpu.gpuWorkLoad(*basic_arguments, GPU_ID)

    # Call function in multi-thread-multi-GPU model
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_count) as executor:
        pointers = []
        for GPU_ID in range(gpu_count):
            pointers.append(
                executor.submit(gpu.gpuWorkLoad, *basic_arguments, val, GPU_ID)
            )

        # for pointer in concurrent.futures.as_completed(pointers):
        #     print(pointer.result())


except Exception as e:
    print(e)
