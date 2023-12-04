import subprocess

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
    try:
        # Run nvidia-smi command to get GPU details in JSON format
        # nvidia_smi_output = subprocess.check_output(['nvidia-smi', '-q', '-x']).decode('utf-8')

        # Alternatively, for a more specific output such as topology, use:
        nvidia_smi_output = subprocess.check_output(['nvidia-smi', 'topo', '-m']).decode('utf-8')
        print(nvidia_smi_output)
       
        return parse_gpu_topology(nvidia_smi_output)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# gpu_topology = """GPU0    GPU1    GPU2    CPU Affinity    NUMA Affinity
# GPU0     X      PIX        NV#                  2              0-1
# GPU1    PIX      X         PIX                  2               0-1
# GPU2    NV#      PIX        X                   2              0-1
# Legend:

# X    = Self
# SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
# NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
# PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
# PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
# PIX  = Connection traversing at most a single PCIe bridge
# NV#  = Connection traversing a bonded set of # NVLinks"""