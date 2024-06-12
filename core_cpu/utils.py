import math

def squared_dist(p1, p2):
    # squared_dist() is much faster than euc_dist()
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def euc_dist(p1, p2):
    # Don't use math.pow -- very slow
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

