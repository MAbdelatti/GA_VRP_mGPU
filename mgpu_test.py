import cupy as cp

with cp.cuda.Device(0):
    x     = cp.zeros((4, 4), dtype=cp.int32)
    #x_ptr = x.data.ptr     # CuPy array pointer
    #print(x_ptr)

with cp.cuda.Device(1):
    print(x)               # prints the array with no error
    x[0, 0] = 99           # set array value from another device

