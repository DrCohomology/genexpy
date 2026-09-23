from genexpy.kernels import rankings as kru
from genexpy.kernels.rankings import MallowsKernel, BordaKernel
from genexpy.utils import rankings as ru

import numpy as np

# r1 = np.array([0, 1, 2, 3])
# r2 = np.array([3, 2, 1, 0])
# na = 4
#
# kernel = MallowsKernel(nu="auto", na = len(r1))
#
# print(kernel(r1, r2))
#
# mink = np.exp(- kernel.nu * (na * (na - 1)) / 2)
#
# print(mink)
#
# out = 0
# for i in range(len(r1)):
#     for j in range(i):
#         out += np.abs(np.sign(r1[i] - r1[j]) - np.sign(r2[i] - r2[j]))
#
# print(na * (na-1) / 2)
# print(out / 2)

r1 = np.array([0, 1, 2, 3])
r2 = np.array([3, 2, 1, 0])
na = 4

kernel = BordaKernel(nu="auto", idx=0, na = len(r1))

print(np.exp(-kernel.nu * (na-1)))

print(kernel(r1, r2))

