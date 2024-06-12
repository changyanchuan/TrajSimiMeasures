import sys
sys.path.append('..')
import numpy as np
import random
import math

from config import Config
from nn.utils.cellspace import CellSpace
from utilities.tool_funcs import truncated_rand
from utilities import tool_funcs


def straight(src):
    return src


def shift(src):
    return [[p[0] + truncated_rand(), p[1] + truncated_rand()] for p in src]


def mask(src, mask_ratio = 0.3):
    # nparray may be faster.
    l = len(src)
    arr = np.array(src)
    mask_idx = np.random.choice(l, int(l * mask_ratio), replace = False)
    return np.delete(arr, mask_idx, 0).tolist()


def add(src):
    l = len(src) - 1
    return


def subset(src, subset_ratio = 0.7):
    l = len(src)
    max_start_idx = l - int(l * subset_ratio)
    start_idx = random.randint(0, max_start_idx)
    end_idx = start_idx + int(l * subset_ratio)
    return src[start_idx: end_idx]


def downsample_and_distort(src, downsample_rate = None, distort_rate = None, distort_dist = 30.0):
    # for CSTRM
    # src: [[lon, lat], [lon, lat], ...]
    
    rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    if downsample_rate == None: 
        downsample_rate = random.choice(rates)
    
    if distort_rate == None: 
        distort_rate = random.choice(rates)
    
    l = len(src)
    arr = np.array(src)
    mask_idx = np.random.choice(l, int(l * downsample_rate), replace = False)
    arr = np.delete(arr, mask_idx, 0)
    
    l = len(arr)
    distort_idx = np.random.choice(l, int(l * distort_rate), replace = False)
    distort_distx = [random.gauss(0, distort_dist) for _ in range(len(distort_idx))]
    distort_disty = [random.gauss(0, distort_dist) for _ in range(len(distort_idx))]
    
    distort_dist = np.zeros( (l, 2) , dtype = arr.dtype)
    np.put(distort_dist, ind = distort_idx * 2, v = distort_distx)
    np.put(distort_dist, ind = distort_idx * 2 + 1, v = distort_disty)
    arr += distort_dist
    
    return arr.tolist()


def get_aug_fn(name: str):
    return {'straight': straight, 'simplify': None, 'shift': shift,
            'mask': mask, 'subset': subset}.get(name, None)


def merc2cell(src, cs: CellSpace):
    # convert and remove consecutive duplicates
    tgt = [cs.get_cellid_by_point(*p) for p in src]
    tgt = [v for i, v in enumerate(tgt) if i == 0 or v != tgt[i-1]]
    return tgt


def merc2cell2(src, cs: CellSpace):
    # convert and remove consecutive duplicates
    tgt = [ (cs.get_cellid_by_point(*p), p) for p in src]
    tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i-1][0]]
    tgt, tgt_p = zip(*tgt)
    return tgt, tgt_p


def generate_spatial_features(src, cs: CellSpace, local_mask_cell = 11):
    # src = [length, 2]
    
    local_mask_sidelen = cs.x_unit * local_mask_cell
    
    tgt = []
    lens = []
    for p1, p2 in tool_funcs.pairwise(src):
        lens.append(tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1]))

    for i in range(1, len(src) - 1):
        dist = (lens[i-1] + lens[i]) / 2
        dist = dist / (local_mask_sidelen / 1.414) # float_ceil(sqrt(2))
        radian = math.pi - math.atan2(src[i-1][0] - src[i][0],  src[i-1][1] - src[i][1]) \
                        + math.atan2(src[i+1][0] - src[i][0],  src[i+1][1] - src[i][1])
        # radian = radian / math.pi
        radian = 1 - abs(radian) / math.pi
        # radian = (radian + math.pi) / (2 * math.pi)
        x = (src[i][0] - cs.x_min) / (cs.x_max - cs.x_min)
        y = (src[i][1] - cs.y_min)/ (cs.y_max - cs.y_min)
        tgt.append( [x, y, dist, radian] )

    x = (src[0][0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[0][1] - cs.y_min)/ (cs.y_max - cs.y_min)
    tgt.insert(0, [x, y, 0.0, 0.0] )
    
    if len(src) >=2 :
        x = (src[-1][0] - cs.x_min) / (cs.x_max - cs.x_min)
        y = (src[-1][1] - cs.y_min)/ (cs.y_max - cs.y_min)
        tgt.append( [x, y, 0.0, 0.0] )
    # tgt = [length, 4]
    return tgt


def traj_len(src):
    length = 0.0
    for p1, p2 in tool_funcs.pairwise(src):
        length += tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1])
    return length

