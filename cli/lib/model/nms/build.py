from __future__ import print_function
import os
import torch
from torch.utils.ffi import create_extension

#this_file = os.path.dirname(__file__)

sources = []
headers = []
defines = []
with_cuda = False


print('Including CUDA code.')
sources += ['src/nms_cuda.c']
headers += ['src/nms_cuda.h']
defines += [('WITH_CUDA', None)]
with_cuda = True

"""
if not torch.cuda.is_available():
    print('Not Including CUDA code.')
    if not hasattr(torch._C, '_cuda_isDriverSufficient'):
        print("issue 1")
    if not torch._C._cuda_isDriverSufficient():
        print("issue 2")
    elif torch._C._cuda_getDeviceCount() <= 0:
        print("issue 3".format(torch._C._cuda_getDeviceCount()))
"""

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['src/nms_cuda_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
print(extra_objects)

ffi = create_extension(
    '_ext.nms',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()
