import os
import subprocess
from mpi4py import MPI
from numba import cuda
from numba.cuda import driver

def getGPUCount():
    cudaDrv = driver.Driver()
    return cudaDrv.get_device_count()

def parse_gpu_topology(gpu_topology):
    lines = gpu_topology.split('\n')
    connectivity = {}
    lineCount = 0
    rows = []

    for line in lines[1:-1]: #ignore the first line
        # Split the line by whitespace and filter out empty strings
        parts = [part for part in line.split() if part]

        # Skip lines that don't contain GPU connectivity data
        if parts and parts[0] != 'Legend:':
            lineCount += 1
            rows.append(parts)
        else:
            break
    # Extract the GPU and its connectivity
    for row in rows:
        gpu, connections = row[0], row[1:]
        connectivity[gpu] = {f"GPU{i}": conn for i, conn in enumerate(connections[:len(rows)])}

    return connectivity

def get_gpu_topology():
    gpu_count = getGPUCount()  # number of available GPUs
    # execute on GPU 0 only:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    try:
        # Run nvidia-smi command to get GPU details:
        nvidia_smi_output = subprocess.check_output(['nvidia-smi', 'topo', '-m']).decode('utf-8')
        # print(nvidia_smi_output)
       
        return (gpu_count, parse_gpu_topology(nvidia_smi_output))

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

        """Example output: GPU0    GPU1    GPU2    CPU Affinity    NUMA Affinity
        GPU0              X      PIX      NV#           2              0-1
        GPU1              PIX      X      PIX           2              0-1
        GPU2              NV#    PIX        X           2              0-1
        Legend:

        X    = Self
        SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
        NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
        PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
        PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
        PIX  = Connection traversing at most a single PCIe bridge
        NV#  = Connection traversing a bonded set of # NVLinks"""

def get_gpu_type():
    # Retrieve GPU type info:
    try:
        # Run nvidia-smi command to get GPU details:
        nvidia_smi_output = subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8')
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    return nvidia_smi_output

def get_gpu_info():
    # GPU info:
    lines = get_gpu_type().split('\n')
    gpu_types = {}

    for line in lines:
        parts = line.split(':')
        if len(parts) > 1 :
            gpu_id = parts[0].strip()
            gpu_name = parts[1].split('(')[0].strip()

        gpu_types[gpu_id] = gpu_name

    gpu_count, gpu_topology = get_gpu_topology()

    # node info:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nodeSize = comm.Get_size()
    data_sent = []
    data_rec = {}
    if rank != 0:
        print('preparing data')
        data_sent = [gpu_count, gpu_types, gpu_topology]
        comm.send(data_sent, dest=0, tag=11)
        print('data sent')
    else:
        if nodeSize > 1:        
            for i in range(nodeSize):
                data_rec[i] = comm.recv(source=i, tag=11)
            
            print(data_rec)
            exit()
            
        nodeList = os.environ.get('SLURM_JOB_NODELIST')

        return(gpu_count, nodeList, nodeSize, gpu_types, gpu_topology)
    # # else:
    # #     return None