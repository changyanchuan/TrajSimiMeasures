# author: yc @ 202301
# ref: 'Sub-trajectory similarity join with obfuscation' - Yanchuan Chang
# ref: @https://github.com/changyanchuan/STS-Join

import math
import numpy as np
from numba import cuda

__all__ = ['cdds']

@cuda.jit(device=True)
def line_slopeintercept(p1, p2):
    # p1, p2: size = [3], endpoints of an edge
    yk = (p2[1] - p1[1]) / (p2[2] - p1[2])
    yb = (p1[1] * p2[2] - p2[1] * p1[2]) / (p2[2] - p1[2])
    xk = (p2[0] - p1[0]) / (p2[2] - p1[2])
    xb = (p1[0] * p2[2] - p2[0] * p1[2]) / (p2[2] - p1[2])
    return xk, xb, yk, yb


@cuda.jit(device=True)
def cdd(e1p1, e1p2, e2p1, e2p2, eps):
    # e1p1, e1p2, e2p1, e2p2: size[3], endpoints of edges e1 and e2
    # eps: distance threshold
    
    if e1p2[2] <= e2p1[2] or e2p2[2] <= e1p1[2]:
        return 0.0
    
    squared_eps = eps*eps

    e1_xk, e1_xb, e1_yk, e1_yb = line_slopeintercept(e1p1, e1p2)
    e2_xk, e2_xb, e2_yk, e2_yb = line_slopeintercept(e2p1, e2p2)

    # distance^2 = _at^2 + _bt + _c (Euclidean Distance^2)
    _a = pow(e1_xk - e2_xk, 2) + pow(e1_yk - e2_yk, 2)
    _b = 2 * ((e1_xk - e2_xk) * (e1_xb - e2_xb) + (e1_yk - e2_yk) * (e1_yb - e2_yb))
    _c = pow(e1_xb - e2_xb, 2) + pow(e1_yb - e2_yb, 2)
    
    if _a == 0 and _b == 0 and _c == 0: # e1 and e2 are on the same line
        return min(e1p2[2], e2p2[2]) - max(e1p1[2], e2p1[2])
    elif _a == 0 and _b == 0:
        if _c <= squared_eps:
            return min(e1p2[2], e2p2[2]) - max(e1p1[2], e2p1[2])
    elif _a == 0:
        _t1 = -1 * _c / _b
        _t2 = (squared_eps - _c) / _b
        t_start = max(e1p1[2], e2p1[2])
        t_end = min(e1p2[2], e2p2[2])

        if _t1 < _t2:
            t_min = max(_t1, t_start)
            t_max = min(_t2, t_end)
            return t_max - t_min if t_max > t_min else 0.0
        elif _t2 < _t1:
            t_min = max(_t2, t_start)
            t_max = min(_t1, t_end)
            return t_max - t_min if t_max > t_min else 0.0
    else: # quatratic
        _c -= squared_eps
        theta = _b * _b - 4 * _a * _c
        if theta > 0:
            theta_sqrt = math.sqrt(theta)
            _t1 = (-1 * theta_sqrt - _b) / 2 / _a
            _t2 = (theta_sqrt - _b) / 2 / _a
            t_min = max(e1p1[2], e2p1[2])
            t_max = min(e1p2[2], e2p2[2])
            t_min = max(_t1, t_min)
            t_max = min(_t2, t_max)
            return t_max - t_min if t_max > t_min else 0.0
    return 0.0
    

@cuda.jit(fastmath=True)
def cdds_opt(trajs, lens, eps, durations):
    # trajs = [batch, 2, seq, 3]
    # lens = [batch, 2]
    # eps: distance threshold
    # durations = [batch, seq]
    
    bid = cuda.blockIdx.x # i-th batch
    subtid = cuda.blockIdx.y # i-th sub-traj of traj1
    pid = subtid * cuda.blockDim.x + cuda.threadIdx.x # i-th point
    
    n1, n2 = lens[bid]
    t2 = trajs[bid, 1, : n2] # size = [t1_len, 3]
    
    duration = 0.0
    
    if pid < n1 - 1:
        e1p1 = trajs[bid, 0, pid]
        e1p2 = trajs[bid, 0, pid+1]
        for j in range(n2 - 1):
            e2p1 = t2[j]
            e2p2 = t2[j+1]
            duration += cdd(e1p1, e1p2, e2p1, e2p2, eps)
        durations[bid, pid] = duration
      


def cdds(t1: np.ndarray, len1: np.ndarray, t2: np.ndarray, len2: np.ndarray, eps,
         threads_per_traj = 64):
    # t1, t2: [batch, seq, 3]
    # len1, len2: [batch]
    # eps: distance threshold
    
    assert t1.shape[0] == t2.shape[0]
    trajs = np.stack([t1, t2], axis = 1) # [batch, 2, seq, 3]
    lens = np.stack([len1, len2], axis = 1) # [batch, 2]
    
    batch_size = t1.shape[0]
    max_trajlen = np.max(lens[:, 0]) # max length of traj1s
    points_per_block = threads_per_traj # default 128
    blocks_per_traj = math.ceil(max_trajlen / points_per_block)
    
    trajs_g = cuda.to_device(trajs) 
    lens_g = cuda.to_device(lens)
    durations_g = cuda.device_array((batch_size, max_trajlen), dtype = np.float64)
    
    with cuda.defer_cleanup():
        cdds_opt[(batch_size, blocks_per_traj), points_per_block](trajs_g, lens_g, eps, durations_g)
        cuda.synchronize()

    durations = durations_g.copy_to_host()
    return durations.sum(axis = -1)

