from libc.math cimport fmax
from libc.math cimport fmin
from libc.math cimport fabs
from libc.math cimport sqrt
from libc.math cimport pow

cimport numpy as np
import numpy as np

def line_slopeintercept(np.ndarray[np.float64_t,ndim=1] p1, np.ndarray[np.float64_t,ndim=1] p2):
    # p1, p2: size = [3], endpoints of an edge
    cdef double yk, yb, xk, xb
    yk = (p2[1] - p1[1]) / (p2[2] - p1[2])
    yb = (p1[1] * p2[2] - p2[1] * p1[2]) / (p2[2] - p1[2])
    xk = (p2[0] - p1[0]) / (p2[2] - p1[2])
    xb = (p1[0] * p2[2] - p2[0] * p1[2]) / (p2[2] - p1[2])
    return xk, xb, yk, yb


def cdd(np.ndarray[np.float64_t,ndim=1] e1p1, np.ndarray[np.float64_t,ndim=1] e1p2,
        np.ndarray[np.float64_t,ndim=1] e2p1, np.ndarray[np.float64_t,ndim=1] e2p2, eps):
    # e1p1, e1p2, e2p1, e2p2: size[3], endpoints of edges e1 and e2
    # eps: distance threshold
    
    if e1p2[2] <= e2p1[2] or e2p2[2] <= e1p1[2] or e1p1[2] == e1p2[2] or e2p1[2] == e2p2[2]:
        return 0.0
    
    cdef double squared_eps = eps*eps

    cdef double e1_xk, e1_xb, e1_yk, e1_yb, e2_xk, e2_xb, e2_yk, e2_yb
    e1_xk, e1_xb, e1_yk, e1_yb = line_slopeintercept(e1p1, e1p2)
    e2_xk, e2_xb, e2_yk, e2_yb = line_slopeintercept(e2p1, e2p2)

    cdef double _a, _b, _c
    # distance^2 = _at^2 + _bt + _c (Euclidean Distance^2)
    _a = pow(e1_xk - e2_xk, 2) + pow(e1_yk - e2_yk, 2)
    _b = 2 * ((e1_xk - e2_xk) * (e1_xb - e2_xb) + (e1_yk - e2_yk) * (e1_yb - e2_yb))
    _c = pow(e1_xb - e2_xb, 2) + pow(e1_yb - e2_yb, 2)
    
    cdef double _t1, _t2, t_start, t_end, t_min, t_max
    cdef double theta, theta_sqrt,
    
    if _a == 0 and _b == 0 and _c == 0: # e1 and e2 are on the same line
        return fmin(e1p2[2], e2p2[2]) - fmax(e1p1[2], e2p1[2])
    elif _a == 0 and _b == 0:
        if _c <= squared_eps:
            return fmin(e1p2[2], e2p2[2]) - fmax(e1p1[2], e2p1[2])
    elif _a == 0:
        _t1 = -1 * _c / _b
        _t2 = (squared_eps - _c) / _b
        t_start = fmax(e1p1[2], e2p1[2])
        t_end = fmin(e1p2[2], e2p2[2])
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
            theta_sqrt = sqrt(theta)
            _t1 = (-1 * theta_sqrt - _b) / 2 / _a
            _t2 = (theta_sqrt - _b) / 2 / _a
            t_min = max(e1p1[2], e2p1[2])
            t_max = min(e1p2[2], e2p2[2])
            t_min = max(_t1, t_min)
            t_max = min(_t2, t_max)
            return t_max - t_min if t_max > t_min else 0.0
    return 0.0
    

def cdds_c(np.ndarray[np.float64_t,ndim=2] t1, np.ndarray[np.float64_t,ndim=2] t2, eps):
    # t1: [seq, 3]
    # t2: [seq, 3]
    # eps: distance threshold
    
    cdef int n1, n2
    cdef double duration = 0.0
    
    n1 = len(t1)
    n2 = len(t2)
    
    cdef i = 0
    cdef j = 0
    cdef np.ndarray[np.float64_t,ndim=1] e1p1, e1p2, e2p1, e2p2
    
    while i < n1 - 1 and j < n2 - 1:
        e1p1, e1p2 = t1[i], t1[i+1]
        e2p1, e2p2 = t2[j], t2[j+1]
        duration += cdd(e1p1, e1p2, e2p1, e2p2, eps)
    
        if e1p2[2] < e2p2[2]:
            i += 1
        elif e2p2[2] < e1p2[2]:
            j += 1
        else:
            i += 1
            j += 1
            
    return duration

