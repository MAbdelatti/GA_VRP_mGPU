#!/usr/bin/env python
# -------- Start of the importing part -----------
import gpu
import readInput
from numba import cuda
from numba.cuda import driver
import concurrent.futures
from timeit import default_timer as timer
import numpy as np
import sys
import gpuGrid
import val
import time

# -------- End of the importing part -----------
# use maximum length in screen output
np.set_printoptions(threshold=sys.maxsize)

# ------------------------- Reading problem data file -------------------------------------------


class vrp:
    def __init__(self, capacity=0, opt=0):
        self.capacity = capacity
        self.opt = opt
        self.nodes = np.zeros((1, 4), dtype=np.float32)

    def addNode(self, label, demand, posX, posY):
        newrow = np.array([label, demand, posX, posY], dtype=np.float32)
        self.nodes = np.vstack((self.nodes, newrow))


# ---------------------------------- Obtain the number of available CUDA GPUs -----------------------------


def getGPUCount():
    cudaDrv = driver.Driver()
    return cudaDrv.get_device_count()


# read problem file and save vrp_capacity, data, opt into basic_arguments list
basic_arguments = readInput()
basic_arguments.append(sys.argv[1])  # filename

val = val.VRP(sys.argv[1], basic_arguments[1].shape[0])
n = int(sys.argv[4])
node_count = basic_arguments[1].shape[0]
totalpopsize = -(-(n * (node_count - 1)) // 1000) * 1000

# Assign the number of GPUs to use:
# gpu_count = 1
gpu_count = getGPUCount()  # Full utilization of GPUs
popsize = min(totalpopsize // gpu_count, int(50e3))

"""Basic arguments to be passed to the kernel:
vrp_capacity, data, known optimal cost, filename, gpu_count, population size, crossover_prob,
mutation_prob, popsize/gpu, crossover_points, blocks, threads_per_block, generations, r_flag"""
basic_arguments.append(gpu_count)  # gpu_count
basic_arguments.append(n)  # n
basic_arguments.append(int(sys.argv[5]))  # crossover_prob
basic_arguments.append(int(sys.argv[6]))  # mutation_prob
basic_arguments.append(popsize)  # popsize
basic_arguments.append(1)  # crossover_points

print("\nFound {} GPUs to Utilize.\n".format(gpu_count))
print(
    "Population size assigned to be {}*n with {} per GPU. Total of {}.  \n".format(
        n, popsize, totalpopsize
    )
)

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
