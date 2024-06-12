# author: yc @ 202301

import math
import numpy as np
from numba import cuda
from typing import Union

from .utils import squared_dist, euc_dist, pad_lists_to_array

__all__ = ['hausdorff']

@cuda.jit(device=True)
def p2s_dist(p, s1, s2, dps1_squared, dps2_squared):
    # p : 1x2 numpy_array
    # s1 : 1x2 numpy_array
    # s2 : 1x2 numpy_array
    # dps1 : euclidean distance between p and s1
    # dps2 : euclidean distance between p and s2
    # dps : euclidean distance between s1 and s2

    px = p[0]
    py = p[1]
    p1x = s1[0]
    p1y = s1[1]
    p2x = s2[0]
    p2y = s2[1]
    segl_squared = squared_dist(s1, s2)
    
    if p1x == p2x and p1y == p2y:
        dist = dps1_squared
    else:
        x_diff = p2x - p1x
        y_diff = p2y - p1y
        u1 = (((px - p1x) * x_diff) + ((py - p1y) * y_diff))
        u = u1 / segl_squared

        if (u < 0.00001) or (u > 1):
            # closest point does not fall within the line segment, take the shorter distance to an endpoint
            dist = min(dps1_squared, dps2_squared)
        else:
            # Intersecting point is on the line, use the formula
            ix = p1x + u * x_diff
            iy = p1y + u * y_diff
            dist = squared_dist(p, (ix, iy))
    return dist
    

@cuda.jit(device=True)
def p2t_dist(p, t, t_len):
    dist = 0.0
    for i in range(t_len - 1):
        s1 = t[i]
        s2 = t[i + 1]
        if i == 0:
            dps1_squared = squared_dist(p, s1)
            dps2_squared = squared_dist(p, s2)
        else:
            dps1_squared = dps2_squared
            dps2_squared = squared_dist(p, s2)

        ps_dist = p2s_dist(p, s1, s2, dps1_squared, dps2_squared)
        
        if i == 0:
            dist = ps_dist
        else:
            if ps_dist < dist:
                dist = ps_dist
            
    return dist
        

@cuda.jit(fastmath=True)
def hausdorff_opt(trajs, lens, p2t_dists):
    # trajs = [batch, 2, seq, 2]
    # lens = [batch, 2]
    # p2t_dists = [batch, 2, seq]
    
    bid = cuda.blockIdx.x # i-th batch
    tid = cuda.blockIdx.y # 0th or 1th traj
    subtid = cuda.blockIdx.z # i-th sub-traj of tid-th-traj
    pid = subtid * cuda.blockDim.x + cuda.threadIdx.x # i-th point
    
    n1, n2 = lens[bid]
    
    if tid == 0 and pid >= n1 or tid == 1 and pid >= n2:
        tid = -1
    
    if tid != -1:
        p = trajs[bid, tid, pid] # size: [2]
        t = trajs[bid, 1-tid] # size: [seq, 2]
        t_len = lens[bid, 1-tid] # float
        pt_dist = p2t_dist(p, t, t_len)
        p2t_dists[bid, tid, pid] = pt_dist
      

# supporting super long trajectories
def hausdorff(t1: Union[np.ndarray, list], len1: Union[np.ndarray, list], 
                t2: Union[np.ndarray, list], len2: Union[np.ndarray, list], 
                threads_per_traj = 64):
    # t1, t2: [batch, seq, 2] for ndarray
    # len1, len2: [batch] for ndarray
    
    assert t1.shape[0] == t2.shape[0]
    trajs = np.stack([t1, t2], axis = 1) # [batch, 2, seq, 2]
    lens = np.stack([len1, len2], axis = 1) # [batch, 2]
    
    batch_size = t1.shape[0]
    max_trajlen = np.max(lens)
    points_per_block = threads_per_traj # default: 128
    blocks_per_traj = math.ceil(max_trajlen / points_per_block)
    
    trajs_g = cuda.to_device(trajs) 
    lens_g = cuda.to_device(lens)
    p2t_dists_g = cuda.device_array((batch_size, 2, max_trajlen), dtype = np.float64)
    
    with cuda.defer_cleanup():
        hausdorff_opt[(batch_size, 2, blocks_per_traj), points_per_block](trajs_g, lens_g, p2t_dists_g)
        cuda.synchronize()

    p2t_dists = p2t_dists_g.copy_to_host()
    t2t_dists = np.sqrt(p2t_dists.max(-1).max(-1))
    
    return t2t_dists

