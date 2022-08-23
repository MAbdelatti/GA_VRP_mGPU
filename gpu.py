#!/usr/bin/env python

from numba import cuda, jit, int32, float32
import cupy as cp
from timeit import default_timer as timer
import numpy as np
import random
import sys
import os
from datetime import datetime
import val
import time
import kernels

np.set_printoptions(threshold=sys.maxsize)
# ------------------------- Main Function ------------------------------------------------------------


def gpuWorkLoad(vrp_capacity, data, opt, filename, gpu_count, n, crossover_prob, mutation_prob,
                popsize, crossover_points,   blocks, threads_per_block, generations, r_flag, val, GPU_ID):

    try:
        global pointers
        global aux_pointers
        pointers = {}
        aux_pointers = {}

        cuda.select_device(GPU_ID)
        print('Start time at GPU {} is: {}'.format(GPU_ID, timer()))

        data_d = cuda.to_device(data)

        print('Data loaded to GPU {}.'.format(GPU_ID))

        # Linear upper triangle of cost table (width=nC2))
        linear_cost_table = cp.zeros(
            (kernels.nCr(data.shape[0], 2)), dtype=np.float32)

        pop_d = cp.ones((popsize, int(1.5*data.shape[0])+2), dtype=np.int32)
        pointers[GPU_ID] = pop_d.data.ptr

        auxiliary_arr = cp.zeros(
            shape=(popsize, pop_d.shape[1]), dtype=cp.int32)

        aux_copy_arr = cp.zeros(
            shape=(popsize, pop_d.shape[1]), dtype=cp.int32)
        aux_pointers[GPU_ID] = aux_copy_arr.data.ptr

        # --------------Calculate the cost table----------------------------------------------
        kernels.calculateLinearizedCost[blocks, threads_per_block](
            data_d, linear_cost_table)
        print('Cost table calculated successfully on GPU {}.'.format(GPU_ID))

        # --------------Initialize population-------------------------------------------------
        kernels.initializePop[blocks, threads_per_block](data_d, pop_d)
        print('Population is initialized successfully on GPU {}.'.format(GPU_ID))

        for individual in pop_d:
            cp.random.shuffle(individual[2:-1])

        kernels.find_duplicates[blocks, threads_per_block](pop_d, r_flag)
        kernels.shift_r_flag[blocks, threads_per_block](r_flag, pop_d)

        kernels.cap_adjust[blocks, threads_per_block](
            r_flag, vrp_capacity, data_d, pop_d)
        kernels.cleanup_r_flag[blocks, threads_per_block](r_flag, pop_d)

        # --------------Calculate fitness----------------------------------------------
        pop_d[:, -1] = 0
        kernels.computeFitness[blocks, threads_per_block](
            linear_cost_table, pop_d, data_d.shape[0])

        # print('computing fitness...')
        # time_list = []
        
        # counter = 0
        # for i in range(100000):
        #     start_time = timer()
        #     pop_d[:, -1] = 0
        #     kernels.computeFitness[blocks, threads_per_block](
        #         linear_cost_table, pop_d, data_d.shape[0])
        #     end_time = timer()
        #     time_list.append(end_time - start_time)
        
        #     counter += 1
        #     if counter%1000 == 0:
        #         print('---- Done {} fitness computations ----'.format(counter))               

        # print('Average time of new function: {} seconds +/- {}'.format(np.mean(time_list), np.std(time_list)))
        # exit(1)

        # print('computing fitness...')
        # time_list = []
        
        # counter = 0
        # for i in range(10000):
        #     start_time = timer()
        #     sum_arr = np.sum(pop_d, axis=1)
        #     for row in range(pop_d.shape[0]):
        #         pop_d[row, -1] = sum_arr[row]
        #     end_time = timer()
        #     time_list.append(end_time - start_time)
        
        #     counter += 1
        #     if counter%100 == 0:
        #         print('---- Done {} fitness computations ----'.format(counter))               

        # print('Average time of new function: {} seconds +/- {}'.format(np.mean(time_list), np.std(time_list)))
        # exit(1)

        # ------------------------Evolve population for some generations------------------------
        parent_idx = cp.ones((popsize, 2), dtype=cp.int32)
        child_d_1 = cp.ones((popsize, pop_d.shape[1]), dtype=cp.int32)
        child_d_2 = cp.ones((popsize, pop_d.shape[1]), dtype=cp.int32)
        cut_idx_d = cp.ones(shape=(pop_d.shape[1]), dtype=cp.int32)

        del_list = [data_d, linear_cost_table, pop_d, auxiliary_arr,
                    parent_idx, child_d_1, child_d_2, cut_idx_d]

        minimum_cost = float('Inf')
        start_time = timer()

        count = 0
        best_sol = 0
        total_time = 0.0
        time_per_loop = 0.0

        while count <= generations:
            if minimum_cost <= opt:
                break

            random_arr_d = cp.arange(
                popsize, dtype=cp.int32).reshape((popsize, 1))
            random_arr_d = cp.repeat(random_arr_d, 4, axis=1)

            for j in range(4):
                cp.random.shuffle(random_arr_d[:, j])

            if aux_copy_arr[0, 0] != 99999:  # If the population was not copied from GPU 0
                # Select parents:
                kernels.selectParents[blocks, threads_per_block](
                    pop_d, random_arr_d, parent_idx)
                kernels.getParentLengths[blocks, threads_per_block](
                    crossover_points, pop_d, auxiliary_arr, parent_idx)
                kernels.generateCutPoints(
                    blocks, threads_per_block, crossover_points, pop_d, popsize, auxiliary_arr)

                random_arr = cp.random.randint(1, 100, (popsize, 1))
                kernels.crossOver[blocks, threads_per_block](
                    random_arr, auxiliary_arr, child_d_1, child_d_2, pop_d, parent_idx, crossover_prob)

                # Performing mutation:
                random_min_max = cp.random.randint(
                    2, pop_d.shape[1]-2, (popsize, 2))
                random_min_max.sort()
                random_no_arr = cp.random.randint(1, 100, (popsize, 1))
                kernels.inverseMutate[blocks, threads_per_block](
                    random_min_max, child_d_1, random_no_arr, mutation_prob)

                random_min_max = cp.random.randint(
                    2, pop_d.shape[1]-2, (popsize, 2))
                random_min_max.sort()
                random_no_arr = cp.random.randint(1, 100, (popsize, 1))
                kernels.inverseMutate[blocks, threads_per_block](
                    random_min_max, child_d_2, random_no_arr, mutation_prob)

                # Adjusting child_1 array:
                kernels.find_duplicates[blocks, threads_per_block](
                    child_d_1, r_flag)
                kernels.findMissingNodes(
                    blocks, threads_per_block, data_d, child_d_1, auxiliary_arr)
                kernels.addMissingNodes[blocks, threads_per_block](
                    r_flag, auxiliary_arr, child_d_1)
                kernels.shift_r_flag[blocks, threads_per_block](
                    r_flag, child_d_1)
                kernels.cap_adjust[blocks, threads_per_block](
                    r_flag, vrp_capacity, data_d, child_d_1)
                kernels.cleanup_r_flag[blocks, threads_per_block](
                    r_flag, child_d_1)

                # Adjusting child_2 array:
                kernels.find_duplicates[blocks, threads_per_block](
                    child_d_2, r_flag)
                kernels.findMissingNodes(
                    blocks, threads_per_block, data_d, child_d_2, auxiliary_arr)
                kernels.addMissingNodes[blocks, threads_per_block](
                    r_flag, auxiliary_arr, child_d_2)
                kernels.shift_r_flag[blocks, threads_per_block](
                    r_flag, child_d_2)
                kernels.cap_adjust[blocks, threads_per_block](
                    r_flag, vrp_capacity, data_d, child_d_2)
                kernels.cleanup_r_flag[blocks, threads_per_block](
                    r_flag, child_d_2)

                # Performing the two-opt optimization and Calculating fitness for child_1 array:
                kernels.reset_to_ones[blocks, threads_per_block](auxiliary_arr)
                kernels.twoOpt[blocks, threads_per_block](
                    child_d_1, auxiliary_arr, linear_cost_table, data_d.shape[0])

                child_d_1[:, -1] = 0
                kernels.computeFitness[blocks, threads_per_block](
                    linear_cost_table, child_d_1, data_d.shape[0])

                # Performing the two-opt optimization and Calculating fitness for child_2 array:
                kernels.reset_to_ones[blocks, threads_per_block](auxiliary_arr)
                kernels.twoOpt[blocks, threads_per_block](
                    child_d_2, auxiliary_arr, linear_cost_table, data_d.shape[0])

                child_d_2[:, -1] = 0
                kernels.computeFitness[blocks, threads_per_block](
                    linear_cost_table, child_d_2, data_d.shape[0])

                # Creating the new population from parents and children:
                kernels.updatePop[blocks, threads_per_block](
                    count, parent_idx, child_d_1, child_d_2, pop_d)
                kernels.elitism(child_d_1, child_d_2, pop_d, popsize)

                pop_d = pop_d[pop_d[:, -1].argsort()]
                if aux_copy_arr[0, 0] != 99999:
                    aux_copy_arr[:, :] = pop_d[:, :]
            else:
                # Receiving population from GPU 0:
                pop_d[:, :] = aux_copy_arr[:, :]
                pop_d[0, 0] = count
                aux_copy_arr[0, 0] = count

            # aux_copy_arr[:, 0] = 99999
            cp.cuda.Device().synchronize()

            # GPU array migration is topology specific:
            if (count+1) % 10000 == 0:
                if gpu_count == 8:  # for hypercube mesh connections like DGX-1
                    # migrate populations at remote GPUs nearby
                    kernels.routePopulation_DGX_1(
                        count, GPU_ID, gpu_count, popsize, pointers, auxiliary_arr, pop_d)
                    cp.cuda.Device().synchronize()  # Sync all GPUs

                    # migrate populations to GPU 0
                    kernels.migratePopulation_DGX_1(
                        GPU_ID, gpu_count, popsize, aux_pointers, auxiliary_arr, pop_d)
                    pop_d = pop_d[pop_d[:, -1].argsort()]

                    cp.cuda.Device().synchronize()  # Sync all GPUs

                    # broadcast updated population at GPU 0 to all GPUs
                    kernels.broadcastPopulation_DGX_1(
                        GPU_ID, aux_pointers, pop_d)
                    pop_d[0, 0] = count

                    cp.cuda.Device().synchronize()  # Sync all GPUs

                elif gpu_count > 1 and gpu_count < 6:  # for P2P-only connection
                    # migrate populations to GPU 0
                    kernels.migratePopulation_P2P(
                        GPU_ID, gpu_count, popsize, aux_pointers, auxiliary_arr, pop_d)
                    pop_d = pop_d[pop_d[:, -1].argsort()]
                    cp.cuda.Device().synchronize()  # Sync all GPUs

                    # broadcast updated population at GPU 0 to all GPUs
                    kernels.broadcastPopulation_P2P(
                        GPU_ID, gpu_count, aux_pointers, pop_d)
                    pop_d[0, 0] = count
                    cp.cuda.Device().synchronize()  # Sync all GPUs

            # Picking best solution:
            best_sol = pop_d[0, :]
            minimum_cost = best_sol[-1]
            worst_cost = pop_d[-1, :][-1]
            delta = worst_cost-minimum_cost
            average = cp.average(pop_d[:, -1])

            if count == 1:
                print('On GPU {}, at first generation, Best: {}, Worst: {}, Delta: {}, Avg: {}'.format(GPU_ID, minimum_cost, worst_cost,
                                                                                                       delta, average))

            elif (count+1) % 10 == 0:
                print('On GPU {}, after {} generations, Best: {}, Worst {}, Delta: {}, Avg: {}'.format(GPU_ID, count+1, minimum_cost, worst_cost,
                                                                                                       delta, average))

            count += 1

        # Final update for the population at GPU 0 for output:
        pop_d = pop_d[pop_d[:, -1].argsort()]
        cp.cuda.Device().synchronize()

        if gpu_count == 8:  # for hypercube mesh connections like DGX-1
            # migrate populations at remote GPUs nearby
            kernels.routePopulation_DGX_1(
                count, GPU_ID, gpu_count, popsize, pointers, auxiliary_arr, pop_d)
            cp.cuda.Device().synchronize()  # Sync all GPUs

            # migrate populations to GPU 0
            kernels.migratePopulation_DGX_1(
                GPU_ID, gpu_count, popsize, aux_pointers, auxiliary_arr, pop_d)
            pop_d = pop_d[pop_d[:, -1].argsort()]
            cp.cuda.Device().synchronize()  # Sync all GPUs

        elif gpu_count > 1 and gpu_count < 6:  # for P2P-only connection
            # migrate populations to GPU 0
            kernels.migratePopulation_P2P(
                GPU_ID, gpu_count, popsize, aux_pointers, auxiliary_arr, pop_d)
            pop_d = pop_d[pop_d[:, -1].argsort()]
            cp.cuda.Device().synchronize()  # Sync all GPUs

        if GPU_ID == 0:
            best_sol = pop_d[0, :]
            minimum_cost = best_sol[-1]
            worst_cost = pop_d[-1, :][-1]
            delta = worst_cost-minimum_cost
            average = cp.average(pop_d[:, -1])

            kernels.showExecutionReport(
                val, filename, opt, count, start_time, best_sol, data, pop_d)

        print('End time at GPU {} is: {}'.format(GPU_ID, timer()))

    except Exception as e:
        print(e)
        kernels.showExecutionReport(
            val, filename, opt, count, start_time, best_sol, data, pop_d)
        print('End time at GPU {} is: {}'.format(GPU_ID, timer()))

    kernels.cleanUp(del_list)
