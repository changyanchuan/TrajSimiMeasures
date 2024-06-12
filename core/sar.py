import math
import numpy as np
import numba
from numba import cuda
from numba import jit

__all__ = ['sar']

@cuda.jit(device=True)
def get_interpolated_point(bid, tid, p1, p2, ts_start, ts_end, timespan,
                           interpolated_points_g, 
                           interpolated_points_len_g):
    # p1, p2: size = [3], endpoints of an edge
    # t: List
    
    yk = (p2[1] - p1[1]) / (p2[2] - p1[2])
    yb = (p1[1] * p2[2] - p2[1] * p1[2]) / (p2[2] - p1[2])
    xk = (p2[0] - p1[0]) / (p2[2] - p1[2])
    xb = (p1[0] * p2[2] - p2[0] * p1[2]) / (p2[2] - p1[2])
    
    idx = interpolated_points_len_g[bid, tid]
    for ts in range(ts_start, ts_end, timespan):
        interpolated_points_g[bid, tid, idx, 0] = xk * ts + xb
        interpolated_points_g[bid, tid, idx, 1] = yk * ts + yb
        interpolated_points_g[bid, tid, idx, 2] = ts
        idx += 1
    interpolated_points_len_g[bid, tid] = idx


@cuda.jit(device=True)
def align_to_timestamps(bid, tid, t, timespan, 
                        interpolated_points_g, 
                        interpolated_points_len_g):
    # t: [seq, 3]
    # timespan: e.g., every 10 seconds
    i = 0
    n = t.shape[0]

    while i < n - 1:
        p1 = t[i]
        p2 = t[i+1]
        
        start_idx = math.ceil(p1[2]/timespan)
        end_idx = math.ceil(p2[2]/timespan)
        if start_idx < end_idx:
            ts_start = start_idx * timespan
            ts_end = end_idx * timespan
            get_interpolated_point(bid, tid, p1, p2, ts_start, ts_end, timespan,
                                   interpolated_points_g, interpolated_points_len_g)
        i += 1
    
    # process the last point
    if t[n-1][2] % timespan == 0:
        _idx = interpolated_points_len_g[bid, tid]
        interpolated_points_g[bid, tid, _idx, 0] = t[n-1][0]
        interpolated_points_g[bid, tid, _idx, 1] = t[n-1][1]
        interpolated_points_g[bid, tid, _idx, 2] = t[n-1][2]
        interpolated_points_len_g[bid, tid] = _idx + 1

    
@cuda.jit(device=True)
def sar_preprocess(bid, t1, t2, time_eps,
           interpolated_points_g, interpolated_points_len_g,
           interpolated_points_idx_g):
    # t1: [seq, 3]
    # t2: [seq, 3]
    # time_eps: length of a time slice, xx seconds
    
    n1 = t1.shape[0]
    n2 = t2.shape[0]
    
    duration_zero = 0
    
    i, j = 0, 0
    i_start, i_end = None, -1
    j_start, j_end = None, -1
    
    # Find overlapped parts
    while i < n1 - 1 and j < n2 - 1:
        e1p1, e1p2 = t1[i], t1[i+1]
        e2p1, e2p2 = t2[j], t2[j+1]
        if e1p2[2] <= e2p1[2] or e2p2[2] <= e1p1[2] or e1p1[2] == e1p2[2] or e2p1[2] == e2p2[2]:
            pass
        else: # overlapped
            if i_start is None: # just once
                i_start = i 
            if j_start is None: # just once
                j_start = j 
                
            i_end = i + 1
            j_end = j + 1
            
        if e1p2[2] < e2p2[2]:
            i += 1
        elif e2p2[2] < e1p2[2]:
            j += 1
        else:
            i += 1
            j += 1
    
    if i_start is None:
        return duration_zero

    # Create the timestamp aligned trajectory for each trajectory input
    align_to_timestamps(bid, 0, t1[i_start: i_end+1], time_eps, 
                              interpolated_points_g, interpolated_points_len_g)
    align_to_timestamps(bid, 1, t2[j_start: j_end+1], time_eps,
                              interpolated_points_g, interpolated_points_len_g)
    
    tt1_sz = interpolated_points_len_g[bid, 0]
    tt2_sz = interpolated_points_len_g[bid, 1]

    if tt1_sz == 0 or tt2_sz == 0:
        return duration_zero
    
    tt1 = interpolated_points_g[bid, 0,  :tt1_sz, :]
    tt2 = interpolated_points_g[bid, 1,  :tt2_sz, :]

    # Strip unoverlapped headings and tails
    tt1_startts, tt1_endts = tt1[0][2], tt1[-1][2]
    tt2_startts, tt2_endts = tt2[0][2], tt2[-1][2]
    
    tt1_startts_idx, tt1_endts_idx = 0, tt1_sz
    tt2_startts_idx, tt2_endts_idx = 0, tt2_sz
    
    if tt1_endts == tt2_endts:
        pass
    elif tt1_endts < tt2_endts:
        cnt = int((tt2_endts - tt1_endts) // time_eps)
        tt2_endts_idx -= min(cnt, tt2_sz)
        tt2_sz = tt2_endts_idx - tt2_startts_idx
    elif tt1_endts > tt2_endts:
        cnt = int((tt1_endts - tt2_endts) // time_eps)
        tt1_endts_idx -= min(cnt, tt1_sz)
        tt1_sz = tt1_endts_idx - tt1_startts_idx
    
    if tt1_sz == 0 or tt2_sz == 0:
        return duration_zero
    
    if tt1_startts == tt2_startts:
        pass
    elif tt1_startts < tt2_startts:
        cnt = int((tt2_startts - tt1_startts) // time_eps)
        tt1_startts_idx += min(cnt, tt1.shape[0])
        tt1_sz = tt1_endts_idx - tt1_startts_idx
    elif tt1_startts > tt2_startts:
        cnt = int((tt1_startts - tt2_startts) // time_eps)
        tt2_startts_idx += min(cnt, tt2.shape[0])
        tt2_sz = tt2_endts_idx - tt2_startts_idx
    
    if tt1_sz == 0 or tt2_sz == 0:
        return duration_zero
    
    assert tt1.shape == tt2.shape
    interpolated_points_idx_g[bid, 0] = tt1_startts_idx
    interpolated_points_idx_g[bid, 1] = tt1_endts_idx
    interpolated_points_idx_g[bid, 2] = tt2_startts_idx
    interpolated_points_idx_g[bid, 3] = tt2_endts_idx
    return 1
    
    
@cuda.jit(device=True)
def sar_compute(bid, subtid, nthread, distance_eps, time_eps, target_length, 
                interpolated_points_g, interpolated_points_idx_g):

    # SAX computation
    tt1_startts_idx = interpolated_points_idx_g[bid, 0]
    tt1_endts_idx = interpolated_points_idx_g[bid, 1]
    tt2_startts_idx = interpolated_points_idx_g[bid, 2]
    tt2_endts_idx = interpolated_points_idx_g[bid, 3]
    tt1 = interpolated_points_g[bid, 0, tt1_startts_idx: tt1_endts_idx, 0:2]
    tt2 = interpolated_points_g[bid, 1, tt2_startts_idx: tt2_endts_idx, 0:2]

    sz = tt1.shape[0]
    cnt = 0
    
    if target_length >= sz:
        for _i in range(sz):
           if math.sqrt(pow(tt2[_i][0] - tt1[_i][0], 2) + pow(tt2[_i][1] - tt1[_i][1], 2)) <= distance_eps:
               cnt += 1
        cnt *= time_eps
        # durations[bid, 0] = cnt
    else:
        _div = int(sz // target_length) # k
        _mod = int(sz % target_length) # m
        
        core_div = int(target_length // nthread)
        core_mod = int(target_length % nthread)
        
        target_i_start = subtid * core_div + min(subtid, core_mod)
        target_i_end = (subtid + 1) * core_div + min(subtid+1, core_mod)
        
        for _i in range(target_i_start, target_i_end):
            _idx_start = _i*_div+min(_i, _mod)
            _idx_end = (_i+1)*_div+min(_i+1, _mod) + 1
            _tt1_x, _tt1_y, _tt2_x, _tt2_y = 0., 0., 0., 0.
            for __i in range(_idx_start, _idx_end):
                _tt1_x += tt1[__i, 0]
                _tt1_y += tt1[__i, 1]
                _tt2_x += tt2[__i, 0]
                _tt2_y += tt2[__i, 1]
            _tt1_x /= (_idx_end - _idx_start) # mean value
            _tt1_y /= (_idx_end - _idx_start)
            _tt2_x /= (_idx_end - _idx_start)
            _tt2_y /= (_idx_end - _idx_start)
            
            if math.sqrt(pow(_tt2_x - _tt1_x, 2) + pow(_tt2_y - _tt1_y, 2)) <= distance_eps:
                cnt += (_idx_end - _idx_start) * time_eps
        # durations[bid, subtid] = cnt
    return cnt
                

@cuda.jit(fastmath=True)
def sar_opt(trajs, lens, distance_eps, time_eps, target_length, 
            interpolated_points_g, interpolated_points_len_g, 
            interpolated_points_idx_g, durations_g):

    # trajs = [batch, 2, seq, 3]
    # lens = [batch, 2]
    # distance_eps: distance threshold
    # time_eps: temporal threshold
    # target_length: i.e., M in original paper
    # durations = [batch]
    
    nthread = cuda.blockDim.x
    bid = cuda.blockIdx.x # i-th batch
    subtid = cuda.threadIdx.x # i-th sub-traj of traj1

    n1, n2 = lens[bid]
    t1 = trajs[bid, 0, : n1] # size = [t1_len, 3]
    t2 = trajs[bid, 1, : n2]
    interpolated_points_len_g[bid, 0] = 0
    interpolated_points_len_g[bid, 1] = 0

    rtn = sar_preprocess(bid, t1, t2, time_eps,
                            interpolated_points_g, interpolated_points_len_g, 
                            interpolated_points_idx_g)
    cuda.syncthreads()
    
    if rtn != 0:
        dur = sar_compute(bid, subtid, nthread, 
                        distance_eps, time_eps, target_length, 
                        interpolated_points_g, interpolated_points_idx_g)
        durations_g[bid, subtid] = dur
      

def sar(t1: np.ndarray, len1: np.ndarray, t2: np.ndarray, len2: np.ndarray, 
        distance_eps, time_eps, target_length, threads_per_traj = 64):
    
    # t1, t2: [batch, seq, 3]
    # len1, len2: [batch]
    # distance_eps: distance threshold
    # time_eps: temporal threshold
    # target_length: i.e., M in original paper
    
    assert t1.shape[0] == t2.shape[0]
    trajs = np.stack([t1, t2], axis = 1) # [batch, 2, seq, 3]
    lens = np.stack([len1, len2], axis = 1) # [batch, 2]
    
    batch_size = t1.shape[0]
    max_trajlen = np.max(lens[:, 0]) # max length of traj1s
    
    ts_range_max_1 = max(trajs[range(batch_size), 0, lens[:, 0]-1, 2] - trajs[range(batch_size), 0, 0, 2])
    ts_range_max_2 = max(trajs[range(batch_size), 1, lens[:, 1]-1, 2] - trajs[range(batch_size), 1, 0, 2])
    ts_range_max = int(max(ts_range_max_1, ts_range_max_2) // time_eps + 2)
    
    trajs_g = cuda.to_device(trajs) 
    lens_g = cuda.to_device(lens)
    durations_g = cuda.device_array((batch_size, threads_per_traj), dtype = np.float64)
    
    interpolated_points_g = cuda.device_array((batch_size, 2, ts_range_max, 3), dtype = np.float32)
    interpolated_points_len_g = cuda.device_array((batch_size, 2), dtype = np.uint32)
    interpolated_points_idx_g = cuda.device_array((batch_size, 4), dtype = np.uint32)
    
    with cuda.defer_cleanup():
        sar_opt[batch_size, threads_per_traj](trajs_g, lens_g, 
                               distance_eps, time_eps, target_length, 
                               interpolated_points_g, 
                               interpolated_points_len_g,
                               interpolated_points_idx_g,
                               durations_g)
        cuda.synchronize()

    durations = durations_g.copy_to_host()
    return durations.sum(axis = -1)

