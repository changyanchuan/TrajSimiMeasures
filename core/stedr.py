# author: yc @ 202301
# ref: 'Towards robust trajectory similarity computation: \
#       Representation‐based spatio‐temporal similarity quantification' - Ziwen Chen

import numpy as np
from numba import cuda

from .utils import squared_dist


__all__ = ['stedr']


@cuda.jit(fastmath=True)
def stedr_opt(trajs, lens, dpmtx, eps, delta, dists):
    # trajs = [batch, 2, seq, 3]
    # traj_len = [batch, 2]
    # dpmtx = [batch, max_trajlen+1, 2], dp vector
    # eps, delta: thresholds
    # dist = [batch]
    
    nthread = cuda.blockDim.x # number of threads for a pair
    
    bid = cuda.blockIdx.x 
    thid = cuda.threadIdx.x # threadID, in range [0, nthread)
    
    n1, n2 = lens[bid, :]
    t1 = trajs[bid, 0, : n1] # size = [t1_len, 3]
    t2 = trajs[bid, 1, : n2] 

    squared_eps = eps*eps

    # initialization
    if thid == 0:
        dpmtx[bid, :, :] = 0
        # no need sync here
    
    niters = n1 + n2 - 1
    for it in range(niters):
        ll_ridx = (it) % 3 # lastlast_row_index for dpmtx
        l_ridx = (it+1) % 3 # last_row_index for dpmtx
        ridx = (it+2) % 3 # current_row_index for dpmtx
        
        pid = thid
        while pid < n2:
            if pid <= it < pid + n1 :
                # we use <= rather than < here by following Ziwen's wwwj 2022 paper.
                subcost = 1  
                if squared_dist(t1[it-pid], t2[pid]) < squared_eps \
                        and abs(t1[it-pid, 2] - t2[pid, 2]) < delta:
                    subcost = 0
                              
                cost = min(dpmtx[bid, pid, ll_ridx] + subcost, 
                           dpmtx[bid, pid, l_ridx] + 1,
                           dpmtx[bid, pid+1, l_ridx] + 1)
                
                dpmtx[bid, pid+1, ridx] = cost
            pid += nthread
        cuda.syncthreads()
        
    if thid == 0:
        dists[bid] = dpmtx[bid, n2, (niters+1)%3] / max(n1, n2) # normalized value
   

def stedr(t1: np.ndarray, len1: np.ndarray, t2: np.ndarray, len2: np.ndarray, eps, delta,
          threads_per_traj = 64):
    # t1, t2: [batch, seq, 3]
    # len1, len2: [batch]
    assert t1.shape[0] == t2.shape[0]
    trajs = np.stack([t1, t2], axis = 1) # [batch, 2, seq, 3]
    lens = np.stack([len1, len2], axis = 1) # [batch, 2]
    
    batch_size = t1.shape[0]
    max_trajlen = lens[:, 1].max()
    
    trajs_g= cuda.to_device(trajs) 
    lens_g = cuda.to_device(lens)
    dpmtx_g = cuda.device_array((batch_size, max_trajlen + 1, 3), dtype = np.uint32)
    dists_g = cuda.device_array(batch_size)

    with cuda.defer_cleanup():
        stedr_opt[batch_size, threads_per_traj](trajs_g, lens_g, dpmtx_g, eps, delta, dists_g)
        cuda.synchronize()

    dists = dists_g.copy_to_host()
    return dists

