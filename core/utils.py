# author: yc @ 202301

import math
import numpy as np
from numba import cuda


@cuda.jit(device=True)
def squared_dist(p1, p2):
    # squared_dist() is much faster than euc_dist()
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


@cuda.jit(device=True)
def euc_dist(p1, p2):
    # Don't use math.pow -- very slow
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# dedicated for 3D list -- trajectory dataset
def pad_lists_to_array(lst):
    dim2 = len(lst[0][0])
    dim1 = max(map(len, lst))
    arr = np.zeros([len(lst), dim1, dim2], dtype = np.float64)
    for i, j in enumerate(lst):
        arr[i][0 : len(j)] = j
    return arr