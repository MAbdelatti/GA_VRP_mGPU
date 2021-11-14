import numpy
import ctypes
from numba import cuda
from numba.cuda.cudadrv import driver, devicearray

gpuptr = ctypes.c_void_p(1234)  # a GPU pointer

mp = driver.MemoryPointer(cuda.current_context(), gpuptr, size=24)

da = devicearray.DeviceNDArray(shape=(8,), strides=(4,), dtype=numpy.dtype(numpy.int32), gpu_data=mp)

print(da.contents)

# To get the ctypes pointer back
print(da.device_ctypes_pointer)

# To get shape, strides, dtype
print(da.shape)
print(da.strides)
print(da.dtype)
