#!/usr/bin/env python
# -------- Start of the importing part -----------
from numba import cuda, jit, int32, float32, int64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numba.cuda import driver
import cupy as cp
from math import pow, hypot, ceil, floor, log
from timeit import default_timer as timer
import numpy as np
import random
import sys
from datetime import datetime
import shutil
import gpuGrid
import val
import time
import os

# ------------------------- Calculating the cost table --------------------------------------
@cuda.jit
def calculateLinearizedCost(data_d, linear_cost_table):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)
    
    for row in range(threadId_row, data_d.shape[0], stride_x):
        for col in range(threadId_col, data_d.shape[0], stride_y):
            if col > row:
                k = int(col - (row*(0.5*row - data_d.shape[0] + 1.5)) - 1)
                linear_cost_table[k] = \
                round(hypot(data_d[row, 2] - data_d[col, 2], data_d[row, 3] - data_d[col, 3]))

# ------------------------- Fitness calculation ---------------------------------------------
@cuda.jit
def fitness_gpu_old(linear_cost_table, pop, n):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        fitnessValue = 0
        pop[row, -1] = 1
        
        if threadId_col == 15:
            for idx in range(1, pop.shape[1]-2):
                i = min(pop[row, idx]-1, pop[row, idx+1]-1)
                j = max(pop[row, idx]-1, pop[row, idx+1]-1)

                if i != j:
                    k = int(j - (i*(0.5*i - n + 1.5)) - 1)
                    fitnessValue += linear_cost_table[k]

            # bit_count     = int((log(fitnessValue) /  log(2)) + 1)
            scaledFitness = fitnessValue  # Scaling the fitness to fit int16
            # scaledFitness = fitnessValue >> bit_count - 16 # Scaling the fitness to fit int16
            pop[row, -1]  = scaledFitness
    
    cuda.syncthreads()

@cuda.jit
def computeFitness(linear_cost_table, pop, n):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        for col in range(threadId_col, pop.shape[1]-2, stride_y):
            i = min(pop[row, col]-1, pop[row, col+1]-1)
            j = max(pop[row, col]-1, pop[row, col+1]-1)

            if i != j:
                k = int(j - (i*(0.5*i - n + 1.5)) - 1)

                cuda.atomic.add(pop, (row, pop.shape[1]-1), linear_cost_table[k])
   
# ------------------------- Refining solutions ---------------------------------------------
@cuda.jit
def find_duplicates_old(pop, r_flag):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        if threadId_col == 15:
            # Detect duplicate nodes:
            for i in range(2, pop.shape[1]-1):
                for j in range(i, pop.shape[1]-1):
                    if pop[row, i] != r_flag and pop[row, j] == pop[row, i] and i != j:
                        pop[row, j] = r_flag

@cuda.jit
def find_missing_nodes_old(data_d, missing_d, pop):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):    
        if threadId_col == 15:
            missing_d[row, threadId_col] = 0        
            # Find missing nodes in the solutions:
            for i in range(1, data_d.shape[0]):
                for j in range(2, pop.shape[1]-1):
                    if data_d[i,0] == pop[row,j]:
                        missing_d[row, i] = 0
                        break
                    else:
                        missing_d[row, i] = data_d[i,0]                  

@cuda.jit
def add_missing_nodes_old(missing_d, pop, r_flag):   
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):       
        if threadId_col == 15:           
            # Add the missing nodes to the solution:
            for k in range(missing_d.shape[1]):
                for l in range(2, pop.shape[1]-1):
                    if missing_d[row, k] != 0 and pop[row, l] == r_flag:
                        pop[row, l] = missing_d[row, k]
                        break                        

@cuda.jit
def find_duplicates(pop, r_flag):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        for col in range(threadId_col, pop.shape[1]-1, stride_y): 
            if col >= 2:
                for j in range(col+1, pop.shape[1]-1):
                    if pop[row, col] != r_flag and pop[row, j] == pop[row, col]:
                        pop[row, j] = r_flag                        

@cuda.jit
def prepareAuxiliary(data_d, missing_d):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, missing_d.shape[0], stride_x):
        for col in range(threadId_col, missing_d.shape[1], stride_y):
            if col < data_d.shape[0]:
                missing_d[row, col] = 0

@cuda.jit
def findExistingNodes(n, pop, exitsing_nodes):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        for col in range(threadId_col, pop.shape[1], stride_y):            
            if pop[row, col] <= n:
                exitsing_nodes[row, pop[row, col]-1] = pop[row, col]            
            
@cuda.jit
def deriveMissingNodes(data_d, missing_d):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, missing_d.shape[0], stride_x):
        for col in range(threadId_col, missing_d.shape[1], stride_y):
            if col < data_d.shape[0]:
                if missing_d[row, col] == 0:
                    missing_d[row, col] = col+1
                else:
                    missing_d[row, col] = 0                

@cuda.jit
def addMissingNodes(r_flag, missing_d, pop):   
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        for col in range(threadId_col, missing_d.shape[1], stride_y):
            if missing_d[row, col] != 0:
                for j in range(2, pop.shape[1]-1):
                    cuda.atomic.compare_and_swap(pop[row,j:], r_flag, missing_d[row, col])
                    if pop[row, j] == missing_d[row, col]:
                        break

@cuda.jit
def shift_r_flag(r_flag, pop):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        if threadId_col == 15:           
            # Shift all r_flag values to the end of the list:        
            for i in range(2, pop.shape[1]-2):
                if pop[row,i] == r_flag:
                    k = i
                    while pop[row,k] == r_flag:
                        k += 1
                    if k < pop.shape[1]-1:
                        pop[row,i], pop[row,k] = pop[row,k], pop[row,i]

@cuda.jit
def cap_adjust(r_flag, vrp_capacity, data_d, pop):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):    
        if threadId_col == 15:
            reqcap = 0.0        # required capacity
            
            # Accumulate capacity:
            i = 1
            while pop[row, i] != r_flag:
                i += 1  
                if pop[row,i] == r_flag:
                    break
            
                if pop[row, i] != 1:
                    reqcap += data_d[pop[row, i]-1, 1] # index starts from 0 while individuals start from 1                
                    if reqcap > vrp_capacity:
                        reqcap = 0
                        # Insert '1' and shift right:
                        new_val = 1
                        rep_val = pop[row, i]
                        for j in range(i, pop.shape[1]-2):
                            pop[row, j] = new_val
                            new_val = rep_val
                            rep_val = pop[row, j+1]
                else:
                    reqcap = 0.0
    cuda.syncthreads()

@cuda.jit
def cleanup_r_flag(r_flag, pop):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        for col in range(threadId_col, pop.shape[1], stride_y):
            if pop[row, col] == r_flag:
                pop[row, col] = 1
    
# ------------------------- Start initializing individuals ----------------------------------------
@cuda.jit
def initializePop(data_d, pop_d):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)
    
    # Generate the individuals from the nodes in data_d:
    for row in range(threadId_row, pop_d.shape[0], stride_x):
        for col in range(threadId_col, data_d.shape[0]+1, stride_y):
            pop_d[row, col] = data_d[col-1, 0]
        
        pop_d[row, 0], pop_d[row, 1] = 1, 1
        
# ------------------------- Start two-opt calculations --------------------------------------------
@cuda.jit
def reset_to_ones(pop):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):    
        for col in range(threadId_col, pop.shape[1], stride_y):
            pop[row, col] = 1   

@cuda.jit
def twoOpt(pop, auxiliary_arr, cost_table, n):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):    
        for col in range(threadId_col, pop.shape[1], stride_y):
            if col+2 < pop.shape[1]:
                # Divide solution into routes, active threads are the firs 1's in the route:
                if pop[row, col] == 1 and pop[row, col+1] != 1 and pop[row, col+2] != 1:
                    route_length = 1
                    while pop[row, col+route_length] != 1 and col+route_length < pop.shape[1]:
                        auxiliary_arr[row, col+route_length] = pop[row, col+route_length]
                        route_length += 1

                    # Now we have auxiliary_arr has the routes to be optimized for every row solution
                    total_cost = 0
                    min_cost =0

                    # compute the route cost:
                    for idx in range(0, route_length):
                        i = min(auxiliary_arr[row,col+idx]-1, auxiliary_arr[row,col+idx+1]-1)
                        j = max(auxiliary_arr[row,col+idx]-1, auxiliary_arr[row,col+idx+1]-1)

                        if i != j:
                            k = int(j - (i*(0.5*i - n + 1.5)) - 1)
                            min_cost += cost_table[k]
                           
                    # So far, the best route is the given one (in auxiliary_arr)
                    improved = True
                    while improved:
                        improved = False
                        for idx_i in range(1, route_length-1):
                                # swap every two pairs
                                auxiliary_arr[row, col+idx_i]  , auxiliary_arr[row, col+idx_i+1] = \
                                auxiliary_arr[row, col+idx_i+1], auxiliary_arr[row, col+idx_i]

                                # check the cost of the arrangement:
                                for idx_j in range(0, route_length):
                                    i = min(auxiliary_arr[row,col+idx_j]-1, auxiliary_arr[row,col+idx_j+1]-1)
                                    j = max(auxiliary_arr[row,col+idx_j]-1, auxiliary_arr[row,col+idx_j+1]-1)

                                    if i != j:
                                        k = int(j - (i*(0.5*i - n + 1.5)) - 1)
                                        total_cost += cost_table[k]
                                
                                if total_cost < min_cost:
                                    min_cost = total_cost
                                    improved = True
                                else:
                                    auxiliary_arr[row, col+idx_i+1], auxiliary_arr[row, col+idx_i]=\
                                    auxiliary_arr[row, col+idx_i]  , auxiliary_arr[row, col+idx_i+1]
                    
                    for idx_k in range(0, route_length):
                        pop[row, col+idx_k] = auxiliary_arr[row, col+idx_k]                        

# --------------------------------- Cross Over part ---------------------------------------------
@cuda.jit  
def selectParents(pop_d, random_arr_d, parent_idx):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):
        for col in range(threadId_col, pop_d.shape[1], stride_y):  
            candid_1_idx = random_arr_d[row, 0]
            candid_2_idx = random_arr_d[row, 1]
            candid_3_idx = random_arr_d[row, 2]
            candid_4_idx = random_arr_d[row, 3]

            # Selecting 1st Parent from binary tournament:
            if pop_d[candid_1_idx, -1] < pop_d[candid_2_idx, -1]:
                parent_idx[row, 0] = candid_1_idx
            else:
                parent_idx[row, 0] = candid_2_idx

            # Selecting 2nd Parent from binary tournament:
            if pop_d[candid_3_idx, -1] < pop_d[candid_4_idx, -1]:
                parent_idx[row, 1] = candid_3_idx
            else:
                parent_idx[row, 1] = candid_4_idx
       
@cuda.jit
def getParentLengths(no_of_cuts, pop_d, auxiliary_arr, parent_idx):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, auxiliary_arr.shape[0], stride_x):
        for col in range(threadId_col, auxiliary_arr.shape[1], stride_y):
            auxiliary_arr[row, col] = 1

        # Calculate the actual length of parents
        if col < pop_d.shape[1]-2:
            if not (pop_d[parent_idx[row, 0], col] == 1 and pop_d[parent_idx[row, 0], col+1] == 1):
                cuda.atomic.add(auxiliary_arr, (row, 2), 1)
                
            if not (pop_d[parent_idx[row, 1], col] == 1 and pop_d[parent_idx[row, 1], col+1] == 1):
                cuda.atomic.add(auxiliary_arr, (row, 3), 1)

            # Minimum length of the two parents
            cuda.atomic.min(auxiliary_arr, (row, 3), auxiliary_arr[row, 2])

            auxiliary_arr[row, 4] = no_of_cuts # k-point crossover

@cuda.jit
def add_cut_points(auxiliary_arr, rng_states):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, auxiliary_arr.shape[0], stride_x):
        if threadId_col == 15:
            no_cuts = auxiliary_arr[row, 4]
            
            for i in range(5, no_cuts+5):
                rnd_val = 0
                
            # Generate unique random numbers as cut indices:
                for j in range(5, no_cuts+5):
                    while rnd_val == 0 or rnd_val == auxiliary_arr[row, j]:
                        rnd = xoroshiro128p_uniform_float32(rng_states, row*auxiliary_arr.shape[1])\
                            *(auxiliary_arr[row, 3] - 2) + 2 # random*(max-min)+min
                        
                        rnd_val = int(rnd)+2            
                
                auxiliary_arr[row, i] = rnd_val
                
            # Sorting the crossover points:
            for i in range(5, no_cuts+5):
                min_index = i
                for j in range(i + 1, no_cuts+5):
                    # Select the smallest value
                    if auxiliary_arr[row, j] < auxiliary_arr[row, min_index]:
                        min_index = j

                auxiliary_arr[row, min_index], auxiliary_arr[row, i] = \
                auxiliary_arr[row, i], auxiliary_arr[row, min_index]
      
@cuda.jit
def crossOver(random_arr, auxiliary_arr, child_d_1, child_d_2, pop_d, parent_idx, crossover_prob):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, auxiliary_arr.shape[0], stride_x):
        for col in range(threadId_col, auxiliary_arr.shape[1] - 1, stride_y):
            if col > 1 and col < child_d_1.shape[1]-1:
                child_d_1[row, col] = pop_d[parent_idx[row, 0], col]
                child_d_2[row, col] = pop_d[parent_idx[row, 1], col]

                if random_arr[row, 0] <= crossover_prob: # Perform crossover with a probability of 0.6
                    no_cuts = auxiliary_arr[row, 4]

                    # Swap from first element to first cut point
                    if col < auxiliary_arr[row, 5]:
                        child_d_1[row, col], child_d_2[row, col] =\
                        child_d_2[row, col], child_d_1[row, col]

                    # For even number of cuts, swap from the last cut point to the end
                    if no_cuts%2 == 0:
                        if col > auxiliary_arr[row, no_cuts+4] and col < child_d_1.shape[1]-1:
                            child_d_1[row, col], child_d_2[row, col] =\
                            child_d_2[row, col], child_d_1[row, col]

                    # Swap in-between elements for 3+ cuts:
                    if no_cuts > 2:
                        for j in range(5, no_cuts+5):
                            cut_idx = auxiliary_arr[row, j]
                            if no_cuts%2 == 0:
                                if j%2==1 and col >= cut_idx and col < auxiliary_arr[row, j+1]:
                                    child_d_1[row, col], child_d_2[row, col] =\
                                    child_d_2[row, col], child_d_1[row, col]
                            
                            elif no_cuts%2 == 1:
                                if j%2==1 and col >= cut_idx and col < auxiliary_arr[row, j+1]:
                                    child_d_1[row, col], child_d_2[row, col] =\
                                    child_d_2[row, col], child_d_1[row, col]


# ------------------------------------Mutation part -----------------------------------------------
@cuda.jit
def inverseMutate(random_min_max, pop, random_no, mutation_prob):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)
    
    for row in range(threadId_row, pop.shape[0], stride_x):
        if random_no[row,0] <= mutation_prob:
            for col in range(threadId_col, pop.shape[1], stride_y):
                start  = random_min_max[row, 0]
                ending = random_min_max[row, 1]
                length = ending - start
                diff   = col - start
                if col >= start and col < start+ceil(length/2):
                    pop[row, col], pop[row, ending-diff] = pop[row, ending-diff], pop[row, col]

# -------------------------- Update population part -----------------------------------------------
@cuda.jit
def updatePop(count, parent_idx, child_d_1, child_d_2, pop_d):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):    
        for col in range(threadId_col, pop_d.shape[1], stride_y):

            if child_d_1[row, -1] <= pop_d[parent_idx[row, 0], -1] and \
            child_d_1[row, -1]    <= pop_d[parent_idx[row, 1], -1] and \
            child_d_1[row, -1]    <= child_d_2[row, -1]:

                pop_d[row, col] = child_d_1[row, col]

            elif child_d_2[row, -1] <= pop_d[parent_idx[row, 0], -1] and \
            child_d_2[row, -1]      <= pop_d[parent_idx[row, 1], -1] and \
            child_d_2[row, -1]      <= child_d_1[row, -1]:

                pop_d[row, col] = child_d_2[row, col]

            elif pop_d[parent_idx[row, 0], -1] <= pop_d[parent_idx[row, 1], -1] and \
            pop_d[parent_idx[row, 0], -1]      <= child_d_1[row, -1] and \
            pop_d[parent_idx[row, 0], -1]      <= child_d_2[row, -1]:

                pop_d[row, col] = pop_d[parent_idx[row, 0], col]

            elif pop_d[parent_idx[row, 1], -1] <= pop_d[parent_idx[row, 0], -1] and \
            pop_d[parent_idx[row, 1], -1]      <= child_d_1[row, -1] and \
            pop_d[parent_idx[row, 1], -1]      <= child_d_2[row, -1]:

                pop_d[row, col] = pop_d[parent_idx[row, 1], col]

            pop_d[row, 0]   = count
                
# ------------------------- Definition of CPU functions ----------------------------------------------   
def findMissingNodes(blocks, threads_per_block, data_d, pop, auxiliary_arr):
    reset_to_ones     [blocks, threads_per_block] (auxiliary_arr)
    prepareAuxiliary  [blocks, threads_per_block] (data_d, auxiliary_arr)
    findExistingNodes [blocks, threads_per_block] (data_d.shape[0], pop, auxiliary_arr)
    deriveMissingNodes[blocks, threads_per_block] (data_d, auxiliary_arr)

def generateCutPoints(blocks, threads_per_block, crossover_points, pop_d, popsize, auxiliary_arr):
    if crossover_points == 1:
        # assign cut points from the middle two quartiles
        auxiliary_arr[:, 5] = cp.random.randint((pop_d.shape[1]//4)*2, (pop_d.shape[1]//4)*3, size=popsize, dtype=cp.int32)
    else:
        rng_states = create_xoroshiro128p_states(popsize*pop_d.shape[1], seed=random.randint(2,2*10**5))
        add_cut_points[blocks, threads_per_block](auxiliary_arr, rng_states)    

def elitism(child_d_1, child_d_2, pop_d, popsize):
    # 5% from parents
    pop_d = pop_d[pop_d[:, -1].argsort()]

    # Sort child 1 & child 2:
    child_d_1 = child_d_1[child_d_1[:,-1].argsort()]
    child_d_2 = child_d_2[child_d_2[:,-1].argsort()]

    # 45% from child 1, and 50% from child 2:
    pop_d[floor(0.05*popsize):floor(0.5*popsize), :] = child_d_1[0:(floor(0.5*popsize)-floor(0.05*popsize)), :]
    pop_d[floor(0.5 *popsize):-1, :]                 = child_d_2[0:(popsize - floor(0.5 *popsize)-1), :]

def showExecutionReport(val, filename, opt, count, start_time, best_sol, data, pop_d):           
    end_time      = timer()
    total_time    = float('{0:.4f}'.format((end_time - start_time)))
    time_per_loop = float('{0:.4f}'.format((end_time - start_time)/(count-1)))

    best_sol     = cp.subtract(best_sol, cp.ones_like(best_sol))
    best_sol[0]  = best_sol[0] + 1
    best_sol[-1] = best_sol[-1] + 1

    print('---------\nProblem: {}, Best known: {}'.format(filename, opt))
    print('Stopped at generation %d, Best cost: %d, from Generation: %d'\
        %(count-1, best_sol[-1], best_sol[0]), end = '\n---------\n')
    print('Best solution:', best_sol, end = '\n---------\n')
    print('Time elapsed:', total_time, 'secs',\
          'Time per loop:', time_per_loop, 'secs', end = '\n---------\n')

    text_out = open('results/'+filename+str(datetime.now())+'.out', 'a')
    print('---------\nProblem:', filename, ', Best known:', opt, file=text_out)    
    print('Stopped at generation %d, Best cost: %d, from Generation: %d'\
        %(count-1, best_sol[-1], best_sol[0]), end = '\n---------\n', file=text_out)
    print('Best solution:', best_sol, end = '\n---------\n', file=text_out)
    print('Time elapsed:', total_time, 'secs, ',\
          'Time per loop:', time_per_loop, 'secs', end = '\n---------\n', file=text_out)
    text_out.close()

    # TODO: uncomment the following lines for the complete runs
    val.read()
    val.costTable()
    val.validate(pop_d, 1)
# -----------------------------------------------------------------------------------------------------
def nCr(n,r):
    f = np.math.factorial
    return int(f(n) / (f(r) * f(n-r)))
# -----------------------------------------------------------------------------------------------------
def cleanUp(del_list):
    del del_list[:]
# -----------------------------------------------------------------------------------------------------
def getGPUCount():
    cudaDrv = driver.Driver()
    return cudaDrv.get_device_count()
# ---------------------------------- Migrate populations from GPUs -------------------------------------------------------------------
def routePopulation_DGX_1(count, GPU_ID, gpu_count, popsize, pointers, auxiliary_arr, pop_d):
# ------------------------+
# GPU 5 >> GPU 1 +
# GPU 6 >> GPU 2 +
# GPU 7 >> GPU 3 +
# ------------------------+
    # Population arrays on all GPUs are already sorted
    if GPU_ID == 1:
        # Copy from GPU 5 >> GPU 1
        auxiliary_arr[:, :]    = 0
        cp.cuda.runtime.memcpyPeer(auxiliary_arr.data.ptr, GPU_ID, pointers[5], 5, pop_d.nbytes)
        pop_d[floor(popsize/gpu_count) : floor(2*popsize/gpu_count), :] = auxiliary_arr[0 : (floor(2*popsize/gpu_count)-floor(popsize/gpu_count)), :]

    if GPU_ID == 2:
        # Copy from GPU 6 >> GPU 2
        auxiliary_arr[:, :]    = 0
        cp.cuda.runtime.memcpyPeer(auxiliary_arr.data.ptr, GPU_ID, pointers[6], 6, pop_d.nbytes)
        pop_d[floor(popsize/gpu_count) : floor(2*popsize/gpu_count), :] = auxiliary_arr[0 : (floor(2*popsize/gpu_count)-floor(popsize/gpu_count)), :]

    if GPU_ID == 3:
        # Copy from GPU 7 >> GPU 3
        auxiliary_arr[:, :]    = 0
        cp.cuda.runtime.memcpyPeer(auxiliary_arr.data.ptr, GPU_ID, pointers[7], 7, pop_d.nbytes)
        pop_d[floor(popsize/gpu_count) : floor(2*popsize/gpu_count), :] = auxiliary_arr[0 : (floor(2*popsize/gpu_count)-floor(popsize/gpu_count)), :]

def migratePopulation_DGX_1(GPU_ID, gpu_count, popsize, pointers, auxiliary_arr, pop_d):
    if GPU_ID == 0:
        # Copy from GPU 4 >> GPU 0
        auxiliary_arr[:, :]    = 0
        cp.cuda.runtime.memcpyPeer(auxiliary_arr.data.ptr, GPU_ID, pointers[4], 4, pop_d.nbytes)
        pop_d[floor(popsize/gpu_count) : floor(2*popsize/gpu_count), :] = auxiliary_arr[0 : floor(2*popsize/gpu_count)-floor(popsize/gpu_count), :]

        # Copy from GPU 1 >> GPU 0
        auxiliary_arr[:, :]    = 0
        cp.cuda.runtime.memcpyPeer(auxiliary_arr.data.ptr, GPU_ID, pointers[1], 1, pop_d.nbytes)
        # pop_d[2*popsize/gpu_count : 4*popsize/gpu_count, :] = auxiliary_arr[0 : 2*popsize/gpu_count, :]
        pop_d[floor(2*popsize/gpu_count) : floor(4*popsize/gpu_count), :] = auxiliary_arr[0 : floor(4*popsize/gpu_count)-floor(2*popsize/gpu_count), :]

        # Copy from GPU 2 >> GPU 0
        auxiliary_arr[:, :]    = 0
        cp.cuda.runtime.memcpyPeer(auxiliary_arr.data.ptr, GPU_ID, pointers[2], 2, pop_d.nbytes)
        # pop_d[4*popsize/gpu_count : 6*popsize/gpu_count, :] = auxiliary_arr[0 : 2*popsize/gpu_count, :]
        pop_d[floor(4*popsize/gpu_count) : floor(6*popsize/gpu_count), :] = auxiliary_arr[0 : floor(6*popsize/gpu_count)-floor(4*popsize/gpu_count), :]

        # Copy from GPU 3 >> GPU 0
        auxiliary_arr[:, :]    = 0
        cp.cuda.runtime.memcpyPeer(auxiliary_arr.data.ptr, GPU_ID, pointers[3], 3, pop_d.nbytes)
        # pop_d[6*popsize/gpu_count : -1, :] = auxiliary_arr[0 : 2*popsize/gpu_count, :]
        pop_d[floor(6*popsize/gpu_count) : -1, :] = auxiliary_arr[0 : (popsize - floor(6*popsize/gpu_count)-1), :]
       
        pop_d = pop_d[pop_d[:,-1].argsort()]

def migratePopulation_P2P(GPU_ID, gpu_count, popsize, pointers, auxiliary_arr, pop_d):
    if GPU_ID == 0:
        for ID_ in range(1, gpu_count):
            # Copy from GPU # ID_ >> GPU 0
            auxiliary_arr[:, :]    = 0
            length = floor(popsize/gpu_count)
            cp.cuda.runtime.memcpyPeer(auxiliary_arr.data.ptr, 0, pointers[ID_], ID_, pop_d.nbytes)
            try:
                pop_d[floor(ID_*length) : floor(ID_*popsize/gpu_count)+length, :] = auxiliary_arr[0: length, :]
            except ValueError:
                pop_d[floor(ID_*length) : -1, :] = auxiliary_arr[0: length, :]

        pop_d = pop_d[pop_d[:,-1].argsort()]
       
def broadcastPopulation_DGX_1(GPU_ID, pointers, pop_d):
    # broadcast updated population at GPU 0 to all GPUs
    if GPU_ID == 0:
        # Copy from GPU 0 >> GPU 4
        cp.cuda.runtime.memcpyPeer(pointers[4], 4, pointers[0], 0, pop_d.nbytes)

        # Copy from GPU 0 >> GPU 1 >> GPU 5
        cp.cuda.runtime.memcpyPeer(pointers[1], 1, pointers[0], 0, pop_d.nbytes)
        cp.cuda.runtime.memcpyPeer(pointers[5], 5, pointers[1], 1, pop_d.nbytes)

        # Copy from GPU 0 >> GPU 2 >> GPU 6
        cp.cuda.runtime.memcpyPeer(pointers[2], 2, pointers[0], 0, pop_d.nbytes)
        cp.cuda.runtime.memcpyPeer(pointers[6], 6, pointers[2], 2, pop_d.nbytes)

        # Copy from GPU 0 >> GPU 3 >> GPU 7
        cp.cuda.runtime.memcpyPeer(pointers[3], 3, pointers[0], 0, pop_d.nbytes)
        cp.cuda.runtime.memcpyPeer(pointers[7], 7, pointers[3], 3, pop_d.nbytes)
     
def broadcastPopulation_P2P(GPU_ID, gpu_count, pointers, pop_d):
    # broadcast updated population at GPU 0 to all GPUs
    if GPU_ID == 0:
        for ID_ in range(1, gpu_count):
            # Copy from GPU 0 >> GPU# ID_
            cp.cuda.runtime.memcpyPeer(pointers[ID_], ID_, pointers[0], 0, pop_d.nbytes)
     
def copyPopulation(pop_d, auxiliary_arr):
    # copy auxiliary_arr values into pop_d
    pop_d[:,:] = auxiliary_arr[:,:]
