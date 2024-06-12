# author: yc @ 202301

import numpy as np
from numba import cuda
from .utils import euc_dist

__all__ = ['dtw']


@cuda.jit(fastmath=True)
def dtw_opt(trajs, lens, dpmtx, dists):
    # trajs = [batch, 2, seq, 2]
    # traj_len = [batch, 2]
    # dpmtx = [batch, max_trajlen+1, 3], dp vector
    # dist = [batch]
    
    nthread = cuda.blockDim.x # number of threads for a pair
    
    bid = cuda.blockIdx.x 
    thid = cuda.threadIdx.x # threadID, in range [0, nthread)
    
    n1, n2 = lens[bid, :]
    t1 = trajs[bid, 0, : n1] # size = [t1_len, 2]
    t2 = trajs[bid, 1, : n2] 

    # initialization
    if thid == 0:
        # didn't use local.array, since the size of the array may exceed 96K
        dpmtx[bid, :, :] = np.inf
        dpmtx[bid, 0, 0] = 0.0
        # no need sync here

    niters = n1 + n2 - 1
    for it in range(niters):
        ll_ridx = (it) % 3 # lastlast_row_index for dpmtx
        l_ridx = (it+1) % 3 # last_row_index for dpmtx
        ridx = (it+2) % 3 # current_row_index for dpmtx
        
        pid = thid
        while pid < n2:
            if pid <= it < pid + n1 :
                cost = euc_dist(t1[it-pid], t2[pid])
                cost += min(dpmtx[bid, pid, ll_ridx], dpmtx[bid, pid, l_ridx], dpmtx[bid, pid+1, l_ridx])
                dpmtx[bid, pid+1, ridx] = cost
            pid += nthread
            
        if it == 0 and thid == 0:
            dpmtx[bid, 0, 0] = np.inf
            
        cuda.syncthreads()
        
    if thid == 0:
        dists[bid] = dpmtx[bid, n2, (niters+1)%3]
   

def dtw(t1: np.ndarray, len1: np.ndarray, t2: np.ndarray, len2: np.ndarray, 
        threads_per_traj = 64):
    # t1, t2: [batch, seq, 2]
    # len1, len2: [batch]
    assert t1.shape == t2.shape
    trajs = np.stack([t1, t2], axis = 1) # [batch, 2, seq, 2]
    lens = np.stack([len1, len2], axis = 1) # [batch, 2]
    
    batch_size = t1.shape[0]
    max_trajlen = lens[:, 1].max()
    
    trajs_g= cuda.to_device(trajs) 
    lens_g = cuda.to_device(lens)
    dpmtx_g = cuda.device_array((batch_size, max_trajlen + 1, 3), dtype = np.float64) # need 3 rows, rather than 2
    dists_g = cuda.device_array(batch_size)

    with cuda.defer_cleanup():
        dtw_opt[batch_size, threads_per_traj](trajs_g, lens_g, dpmtx_g, dists_g)
        cuda.synchronize()

    dists = dists_g.copy_to_host()
    return dists

