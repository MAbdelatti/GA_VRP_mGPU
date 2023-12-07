import numpy as np
import sys

# ------------------------- Reading problem data file -------------------------------------------

class vrp:
    def __init__(self, capacity=0, opt=0):
        self.capacity = capacity
        self.opt = opt
        self.nodes = np.zeros((1, 4), dtype=np.float32)

    def addNode(self, label, demand, posX, posY):
        newrow = np.array([label, demand, posX, posY], dtype=np.float32)
        self.nodes = np.vstack((self.nodes, newrow))

def readInput():
    # Create VRP object:
    vrpManager = vrp()
    # First reading the VRP from the input #
    fo = open(sys.argv[1], "r")
    lines = fo.readlines()
    for i, line in enumerate(lines):
        while line.upper().startswith("COMMENT"):
            if len(sys.argv) <= 3:
                inputs = line.split()
                if inputs[-1][:-1].isnumeric():
                    vrpManager.opt = np.int32(inputs[-1][:-1])
                    break
                else:
                    try:
                        vrpManager.opt = float(inputs[-1][:-1])
                    except:
                        print("\nNo optimal value detected, taking optimal as 0.0")
                        vrpManager.opt = 0.0
                    break
            else:
                vrpManager.opt = np.int32(sys.argv[3])
                # print("\nManual optimal value entered: %d" % vrpManager.opt)
                break

        # Validating positive non-zero capacity
        if vrpManager.opt < 0:
            print(sys.stderr, "Invalid input: optimal value can't be negative!")
            exit(1)
            break

        while line.upper().startswith("CAPACITY"):
            inputs = line.split()
            try:
                vrpManager.capacity = np.float32(inputs[2])
            except IndexError:
                vrpManager.capacity = np.float32(inputs[1])
                # Validating positive non-zero capacity
            if vrpManager.capacity <= 0:
                print(
                    sys.stderr,
                    "Invalid input: capacity must be neither negative nor zero!",
                )
                exit(1)
            break
        while line.upper().startswith("NODE_COORD_SECTION"):
            i += 1
            line = lines[i]
            while not (line.upper().startswith("DEMAND_SECTION") or line == "\n"):
                inputs = line.split()
                vrpManager.addNode(
                    np.int16(inputs[0]),
                    0.0,
                    np.float32(inputs[1]),
                    np.float32((inputs[2])),
                )

                i += 1
                line = lines[i]
                while line == "\n":
                    i += 1
                    line = lines[i]
                    if line.upper().startswith("DEMAND_SECTION"):
                        break
                if line.upper().startswith("DEMAND_SECTION"):
                    i += 1
                    line = lines[i]
                    while not (line.upper().startswith("DEPOT_SECTION")):
                        inputs = line.split()
                        # Validating demand not greater than capacity
                        if float(inputs[1]) > vrpManager.capacity:
                            print(
                                sys.stderr,
                                "Invalid input: the demand of the node %s is greater than the vehicle capacity!"
                                % vrpManager.nodes[0],
                            )
                            exit(1)
                        if float(inputs[1]) < 0:
                            print(
                                sys.stderr,
                                "Invalid input: the demand of the node %s cannot be negative!"
                                % vrpManager.nodes[0],
                            )
                            exit(1)
                        vrpManager.nodes[int(inputs[0])][1] = float(inputs[1])
                        i += 1
                        line = lines[i]
                        while line == "\n":
                            i += 1
                            line = lines[i]
                            if line.upper().startswith("DEPOT_SECTION"):
                                break
                        if line.upper().startswith("DEPOT_SECTION"):
                            vrpManager.nodes = np.delete(vrpManager.nodes, 0, 0)
                            # print("Done.")
                            return [
                                vrpManager.capacity,
                                vrpManager.nodes,
                                vrpManager.opt,
                            ]