# ref1: A Symbolic Representation of Time Series, with Implications for Streaming Algorithms
# ref2: Efficient Trajectory Joins using Symbolic Representations

# yanchuan: Symbolic Approximation Representation: SAR, not same to SAX

from libc.math cimport fmax
from libc.math cimport fmin
from libc.math cimport pow
from libc.math cimport ceil
from libc.math cimport floor

cimport numpy as np
import numpy as np

def get_interpolated_point(np.ndarray[np.float64_t,ndim=1] p1, 
                           np.ndarray[np.float64_t,ndim=1] p2, 
                           ts):

    # p1, p2: size = [3], endpoints of an edge
    cdef double yk, yb, xk, xb
    yk = (p2[1] - p1[1]) / (p2[2] - p1[2])
    yb = (p1[1] * p2[2] - p2[1] * p1[2]) / (p2[2] - p1[2])
    xk = (p2[0] - p1[0]) / (p2[2] - p1[2])
    xb = (p1[0] * p2[2] - p2[0] * p1[2]) / (p2[2] - p1[2])
    
    return [(xk * _t + xb, yk * _t + yb, _t) for _t in ts]


def align_to_timestamps(np.ndarray[np.float64_t,ndim=2] t, timespan):
    # t: [seq, 3]
    # timespan: e.g., every 10 seconds

    cdef int i = 0
    cdef int n = t.shape[0]
    cdef list new_t = []
    
    cdef np.ndarray[np.float64_t,ndim=1] p1, p2
    cdef list ts, interpolated_points

    while i < n - 1:
        p1 = t[i]
        p2 = t[i+1]
        
        ts = list(range(int(ceil(p1[2]/timespan)), int(ceil(p2[2]/timespan)), 1))
        
        if len(ts) > 0:
            ts = [_ts*timespan for _ts in ts]
            interpolated_points = get_interpolated_point(p1, p2, ts)
            new_t += interpolated_points

        i += 1

    # process the last point
    if t[n-1][2] % timespan == 0:
        # print(type(t))
        new_t += [t[n-1]]

    if len(new_t):
        return np.array(new_t)
    else:
        return np.empty([0,0])
    

def sar_c(np.ndarray[np.float64_t,ndim=2] t1, np.ndarray[np.float64_t,ndim=2] t2, 
          distance_eps, time_eps, target_length):

    # t1: [seq, 3]
    # t2: [seq, 3]
    # distance_eps: distance threshold, xx meters
    # time_eps: length of a time slice, xx seconds
    # target_length: length of the target trajectory
    
    cdef int n1, n2
    n1 = t1.shape[0]
    n2 = t2.shape[0]
    
    cdef double duration_zero = 0.0
    
    cdef i = 0
    cdef j = 0
    cdef i_start = -1
    cdef i_end = -1
    cdef j_start = -1
    cdef j_end = -1
    
    cdef np.ndarray[np.float64_t,ndim=1] e1p1, e1p2, e2p1, e2p2
    
    # Find overlapped parts
    while i < n1 - 1 and j < n2 - 1:
        e1p1, e1p2 = t1[i], t1[i+1]
        e2p1, e2p2 = t2[j], t2[j+1]
        if e1p2[2] <= e2p1[2] or e2p2[2] <= e1p1[2] or e1p1[2] == e1p2[2] or e2p1[2] == e2p2[2]:
            pass
        else: # overlapped
            if i_start < 0: # just once
                i_start = i 
            if j_start < 0: # just once
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
    
    if i_start < 0:
        return duration_zero
    
    # Create the timestamp aligned trajectory for each trajectory input
    cdef np.ndarray[np.float64_t, ndim=2] tt1, tt2
    tt1 = align_to_timestamps(  t1[i_start: i_end+1], time_eps  )
    tt2 = align_to_timestamps(  t2[j_start: j_end+1], time_eps  )
    
    if tt1.shape[0] == 0 or tt2.shape[0] == 0:
        return duration_zero

    # Strip unoverlapped headings and tails
    cdef double tt1_startts, tt1_endts,  tt2_startts, tt2_endts
    tt1_startts, tt1_endts = tt1[0][2], tt1[-1][2]
    tt2_startts, tt2_endts = tt2[0][2], tt2[-1][2]

    cdef int cnt, _idx_start, _idx_end
    cdef list _idx_range
    if tt1_endts == tt2_endts:
        pass
    elif tt1_endts < tt2_endts:
        cnt = int((tt2_endts - tt1_endts) / time_eps)
        _idx_end = tt2.shape[0]
        _idx_start = int(fmax(tt2.shape[0] - cnt, 0))
        _idx_range = list(range(_idx_start, _idx_end))
        tt2 = np.delete(tt2, _idx_range, 0)
    elif tt1_endts > tt2_endts:
        cnt = int((tt1_endts - tt2_endts) / time_eps)
        _idx_end = tt1.shape[0]
        _idx_start = int(fmax(tt1.shape[0] - cnt, 0))
        _idx_range = list(range(_idx_start, _idx_end))
        tt1 = np.delete(tt1, _idx_range, 0)
    
    if tt1.shape[0] == 0 or tt2.shape[0] == 0:
        return duration_zero
    
    if tt1_startts == tt2_startts:
        pass
    elif tt1_startts < tt2_startts:
        cnt = int((tt2_startts - tt1_startts) / time_eps)
        _idx_start = 0
        _idx_end = int(fmin(cnt, tt1.shape[0]))
        _idx_range = list(range(_idx_start, _idx_end))
        tt1 = np.delete(tt1, _idx_range, 0)
    elif tt1_startts > tt2_startts:
        cnt = int((tt1_startts - tt2_startts) / time_eps)
        _idx_start = 0
        _idx_end = int(fmin(cnt, tt2.shape[0]))
        _idx_range = list(range(_idx_start, _idx_end))
        tt2 = np.delete(tt2, _idx_range, 0)
    
    if tt1.shape[0] == 0 or tt2.shape[0] == 0:
        return duration_zero
    
    tt1 = tt1[:, 0:2]
    tt2 = tt2[:, 0:2]
    # SAX computation
    cdef int sz = tt1.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] _sq_dist
    
    if target_length >= sz:
        cnt = int(np.sum(np.sqrt(np.sum(np.power(tt2 - tt1, 2), axis = 1)) <= distance_eps)) * time_eps
    else:
        splits = np.array_split(range(sz), target_length)
        cnt = 0

        for s in splits:
            _idx_start, _idx_end = s[0], s[-1] + 1
            _sq_dist = np.power( np.mean(tt2[_idx_start: _idx_end], axis=0) - np.mean(tt1[_idx_start: _idx_end], axis=0) , 2)
            cnt = cnt + int(np.sum(np.sqrt(np.sum(_sq_dist)) <= distance_eps)) * (_idx_end - _idx_start) * time_eps
    return cnt



   