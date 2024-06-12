
from libc.math cimport fmax
from libc.math cimport fmin
from libc.math cimport fabs
from libc.math cimport sqrt

cimport numpy as np
import numpy as np


def euc_dist_c(x1, y1, x2, y2):
    return sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) )


def stlcss_c(np.ndarray[np.float64_t,ndim=2] t1, np.ndarray[np.float64_t,ndim=2] t2, eps, delta):
    cdef int n1, n2, i, j
    cdef np.ndarray[np.float64_t,ndim=2] dpmtx
    cdef double x1, y1, ts1, x2, y2, ts2, rtn

    n1 = len(t1)
    n2 = len(t2)

    dpmtx=np.zeros((n1+1,n2+1))

    for i from 0 <= i < n1:
        for j from 0 <= j < n2:
            x1=t1[i,0]
            y1=t1[i,1]
            ts1=t1[i,2]
            x2=t2[j,0]
            y2=t2[j,1]
            ts2=t2[j,2]
            
            if euc_dist_c(x1, y1, x2, y2)<eps and fabs(ts1-ts2)<delta:
                dpmtx[i+1,j+1] = dpmtx[i,j] + 1
            else:
                dpmtx[i+1,j+1] = fmax(dpmtx[i+1,j], dpmtx[i,j+1])
    rtn = 1-float(dpmtx[n1,n2])/fmin(n1,n2)
    return rtn