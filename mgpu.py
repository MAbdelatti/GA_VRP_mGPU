#!/usr/bin/env python
# -------- Start of the importing part -----------
import gpu
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
np.set_printoptions(threshold=sys.maxsize) # use maximum length in screen output

# ------------------------- Reading problem data file -------------------------------------------
class vrp():
    def __init__(self, capacity=0, opt=0):
        self.capacity = capacity
        self.opt = opt
        self.nodes = np.zeros((1,4), dtype=np.float32)

    def addNode(self, label, demand, posX, posY):
        newrow = np.array([label, demand, posX, posY], dtype=np.float32)
        self.nodes = np.vstack((self.nodes, newrow))

def readInput():

    # Create VRP object:
    vrpManager = vrp()
    # First reading the VRP from the input #
    print('Reading data file...', end=' ')
    fo = open(sys.argv[1],"r")
    lines = fo.readlines()
    for i, line in enumerate(lines):       
        while line.upper().startswith('COMMENT'):
            if len(sys.argv) <= 3:
                inputs = line.split()
                if inputs[-1][:-1].isnumeric():
                    vrpManager.opt = np.int32(inputs[-1][:-1])
                    break
                else:
                    try:
                        vrpManager.opt = float(inputs[-1][:-1])
                    except:
                        print('\nNo optimal value detected, taking optimal as 0.0')
                        vrpManager.opt = 0.0
                    break
            else:
                vrpManager.opt = np.int32(sys.argv[3])
                print('\nManual optimal value entered: %d'%vrpManager.opt)
                break

        # Validating positive non-zero capacity
        if vrpManager.opt < 0:
            print(sys.stderr, "Invalid input: optimal value can't be negative!")
            exit(1)
            break

        while line.upper().startswith('CAPACITY'):
            inputs = line.split()
            try:
                vrpManager.capacity = np.float32(inputs[2])
            except IndexError:
                vrpManager.capacity = np.float32(inputs[1])
			# Validating positive non-zero capacity
            if vrpManager.capacity <= 0:
                print(sys.stderr, 'Invalid input: capacity must be neither negative nor zero!')
                exit(1)
            break       
        while line.upper().startswith('NODE_COORD_SECTION'):
            i += 1
            line = lines[i]
            while not (line.upper().startswith('DEMAND_SECTION') or line=='\n'):
                inputs = line.split()
                vrpManager.addNode(np.int16(inputs[0]), 0.0, np.float32(inputs[1]), np.float32((inputs[2])))

                i += 1
                line = lines[i]
                while (line=='\n'):
                    i += 1
                    line = lines[i]
                    if line.upper().startswith('DEMAND_SECTION'): break 
                if line.upper().startswith('DEMAND_SECTION'):
                    i += 1
                    line = lines[i] 
                    while not (line.upper().startswith('DEPOT_SECTION')):                  
                        inputs = line.split()
						# Validating demand not greater than capacity
                        if float(inputs[1]) > vrpManager.capacity:
                            print(sys.stderr,
							'Invalid input: the demand of the node %s is greater than the vehicle capacity!' % vrpManager.nodes[0])
                            exit(1)
                        if float(inputs[1]) < 0:
                            print(sys.stderr,
                            'Invalid input: the demand of the node %s cannot be negative!' % vrpManager.nodes[0])
                            exit(1)                            
                        vrpManager.nodes[int(inputs[0])][1] =  float(inputs[1])
                        i += 1
                        line = lines[i]
                        while (line=='\n'):
                            i += 1
                            line = lines[i]
                            if line.upper().startswith('DEPOT_SECTION'): break
                        if line.upper().startswith('DEPOT_SECTION'):
                            vrpManager.nodes = np.delete(vrpManager.nodes, 0, 0) 
                            print('Done.')
                            return [vrpManager.capacity, vrpManager.nodes, vrpManager.opt]

# ---------------------------------- Obtain the number of available CUDA GPUs ----------------------------- 
def getGPUCount():
    cudaDrv = driver.Driver()
    return cudaDrv.get_device_count()

basic_arguments = readInput() # read problem file and save vrp_capacity, data, opt into basic_arguments list
basic_arguments.append(sys.argv[1]) #filename

val                     = val.VRP(sys.argv[1], basic_arguments[1].shape[0])
n                       = int(sys.argv[4])
node_count              = basic_arguments[1].shape[0]
totalpopsize            = -(-(n*(node_count - 1))//1000)*1000

# Assign the number of GPUs to use:
gpu_count               = 1
# gpu_count               = getGPUCount() # Full utilization of GPUs
popsize                 = min(totalpopsize//gpu_count, int(50e3))

basic_arguments.append(gpu_count)         # gpu_count 
basic_arguments.append(n)                 # n
basic_arguments.append(int(sys.argv[5]))  # crossover_prob
basic_arguments.append(int(sys.argv[6]))  # mutation_prob 
basic_arguments.append(popsize) # popsize
basic_arguments.append(1) # crossover_points

print("\nFound {} GPUs to Utilize.\n".format(gpu_count))
print('Population size assigned to be {}*n with {} per GPU. Total of {}.  \n'.format(n, popsize, totalpopsize))

# GPU grid configurations:
grid               = gpuGrid.GRID()
blocks_x, blocks_y = grid.blockAlloc(node_count, float(n))
tpb_x, tpb_y       = grid.threads_x, grid.threads_y
basic_arguments.append((blocks_x, blocks_y))  # blocks
basic_arguments.append((tpb_x, tpb_y))        # threads_per_block

print(grid)  
try:
    generations = int(sys.argv[2])
except:
    print('No generation limit provided, taking 2000 generations...')
    generations = 2000

basic_arguments.append(generations)

r_flag = 99999   # A flag for removal/replacement
basic_arguments.append(r_flag)

try:
    # Call function in single-thread-multi-GPU model
    # for GPU_ID in range(gpu_count):
    #         gpu.gpuWorkLoad(*basic_arguments, GPU_ID)
    
    # Call function in multi-thread-multi-GPU model
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_count) as executor:        
        pointers = []
        for GPU_ID in range(gpu_count):
            pointers.append(executor.submit(gpu.gpuWorkLoad, *basic_arguments, val, GPU_ID))
        
        # for pointer in concurrent.futures.as_completed(pointers):
        #     print(pointer.result())

        
except error as e:
    print(e)