# author: yc @ 202301

import numpy as np
from numba import cuda
from .utils import euc_dist

__all__ = ['erp']


@cuda.jit(fastmath=True)
def erp_opt(trajs, lens, dpmtx, gpoint, gpdists_g, dists):
    # trajs = [batch, 2, seq, 2]
    # traj_len = [batch, 2]
    # dpmtx = [batch, max_trajlen+1, 2], dp vector
    # gpoint = [2], a 2D point
    # gpdists_g = [batch, 2, max_trajlen], distance between g and trajectory points
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
        dpmtx[bid, :, :] = 0.0
        
    pid = thid
    while pid < n1:
        gpdists_g[bid, 0, pid] = euc_dist(gpoint, t1[pid])
        pid += nthread

    pid = thid
    while pid < n2:
        gpdists_g[bid, 1, pid] = euc_dist(gpoint, t2[pid])
        pid += nthread
        
    cuda.syncthreads()
    
    t1_gpdists_sum = 0
    for i in range(n1):
        t1_gpdists_sum += gpdists_g[bid, 0, i]
    t2_gpdists_sum = 0
    for i in range(n2):
        t2_gpdists_sum += gpdists_g[bid, 1, i]
    # no need to sync
    
    niters = n1 + n2 - 1
    for it in range(niters):
        ll_ridx = (it) % 3 # lastlast_row_index for dpmtx
        l_ridx = (it+1) % 3 # last_row_index for dpmtx
        ridx = (it+2) % 3 # current_row_index for dpmtx
        
        pid = thid
        while pid < n2:
            if pid <= it < pid + n1 :
                
                if pid > 0 and it-pid > 0:
                    dist_left = dpmtx[bid, pid, l_ridx]
                    dist_upper = dpmtx[bid, pid+1, l_ridx]
                    dist_upperleft = dpmtx[bid, pid, ll_ridx]
                else:
                    if pid > 0:
                        dist_left = dpmtx[bid, pid, l_ridx]
                        dist_upper = t2_gpdists_sum
                        dist_upperleft = t2_gpdists_sum
                    elif it-pid > 0:
                        dist_left = t1_gpdists_sum
                        dist_upper = dpmtx[bid, pid+1, l_ridx]
                        dist_upperleft = t1_gpdists_sum
                    else:
                        dist_left = t1_gpdists_sum
                        dist_upper = t2_gpdists_sum
                        dist_upperleft = dpmtx[bid, pid, ll_ridx]
                    
                cost_left = dist_left + gpdists_g[bid, 1, pid]
                cost_upper = dist_upper + gpdists_g[bid, 0, it-pid]
                cost_upperleft = dist_upperleft + euc_dist(t1[it-pid], t2[pid])
                cost = min(cost_left, cost_upper, cost_upperleft)
                
                dpmtx[bid, pid+1, ridx] = cost
            pid += nthread
            
        cuda.syncthreads()
            
    if thid == 0:
        dists[bid] = dpmtx[bid, n2, (niters+1)%3]
   

def erp(t1: np.ndarray, len1: np.ndarray, t2: np.ndarray, len2: np.ndarray, gpoint = None, 
        threads_per_traj = 64):
    # t1, t2: [batch, seq, 2]
    # len1, len2: [batch]
    # gpoint: [2]
    
    assert t1.shape[0] == t2.shape[0]
    trajs = np.stack([t1, t2], axis = 1) # [batch, 2, seq, 2]
    lens = np.stack([len1, len2], axis = 1) # [batch, 2]
    gpoint = np.asarray([0.0, 0.0], np.float64) if gpoint == None else gpoint
    
    batch_size = t1.shape[0]
    max_trajlen = lens[:, 1].max()
    
    trajs_g= cuda.to_device(trajs) 
    lens_g = cuda.to_device(lens)
    dpmtx_g = cuda.device_array((batch_size, max_trajlen + 1, 3), dtype = np.float64)
    gpoint_g = cuda.to_device(gpoint)
    gpdists_g = cuda.device_array((batch_size, 2, max_trajlen), dtype = np.float64)
    dists_g = cuda.device_array(batch_size)

    with cuda.defer_cleanup():
        erp_opt[batch_size, threads_per_traj](trajs_g, lens_g, dpmtx_g, gpoint_g, gpdists_g, dists_g)
        cuda.synchronize()

    dists = dists_g.copy_to_host()
    return dists

