# code ref: https://github.com/yaodi833/NeuTraj
# all methods are generally same to diyao's original implementation.
# I optimized some original implementions, since the original implementation is inefficient
# Also, fit their code to our data format -- yc
import sys
sys.path.append('..')
import time
import logging
import numpy as np

from config import Config as Config


class Preprocesser(object):
    def __init__(self, delta = 0.005, lat_range = [1,2], lon_range = [1,2]):
        self.delta = delta
        self.lat_range = lat_range
        self.lon_range = lon_range
        self._init_grid_hash_function()

    def _init_grid_hash_function(self):
        dXMax, dXMin, dYMax, dYMin = self.lon_range[1], self.lon_range[0], self.lat_range[1], self.lat_range[0]
        x  = self._frange(dXMin, dXMax, self.delta)
        y  = self._frange(dYMin, dYMax, self.delta)
        self.x = x # list
        self.y = y

    def _frange(self, start, end=None, inc=None):
        "A range function, that does accept float increments..."
        if end == None:
            end = start + 0.0
            start = 0.0
        if inc == None:
            inc = 1.0
        L = []
        while 1:
            next_ = start + len(L) * inc
            if inc > 0 and next_ >= end:
                break
            elif inc < 0 and next_ <= end:
                break
            L.append(next_)
        return L

    # return grid value in grid space and grid index
    def get_grid_index(self, tuple): #(lon, lat)
        test_tuple = tuple
        test_x,test_y = test_tuple[0],test_tuple[1]
        x_grid = int ((test_x-self.lon_range[0])/self.delta)
        y_grid = int ((test_y-self.lat_range[0])/self.delta)
        index = (y_grid)*(len(self.x)) + x_grid
        return x_grid, y_grid, index

    # convert original traj to grid traj, and meanwhile discard duplicated points
    # when isCoordinate = true: return the sequence of satisfied original points x, y 
    # when isCoordinate = false: return grid index 
    def traj2grid_seq(self, traj = [], isCoordinate = False):
        grid_traj = []
        for r in traj: # trajs = [[lon, lat], [lon, lat], ...] 
            x_grid, y_grid, gindex = self.get_grid_index((r[0], r[1]))
            grid_traj.append(gindex)

        privious = None
        hash_traj = []
        for i, gindex in enumerate(grid_traj):
            if privious == None:
                privious = gindex
                if isCoordinate == False:
                    hash_traj.append(gindex)
                elif isCoordinate == True:
                    hash_traj.append(traj[i])
            else:
                if gindex == privious: # grid coordinates duplicated
                    pass
                else:
                    if isCoordinate == False:
                        hash_traj.append(gindex)
                    elif isCoordinate == True:
                        hash_traj.append(traj[i])
                    privious = gindex
        return hash_traj

    def _traj2grid_preprocess(self, traj_feature_map, isCoordinate = False):
        trajs_hash = []
        # trajs_keys = traj_feature_map.keys()
        # for traj_key in trajs_keys:
            # traj = traj_feature_map[traj_key]
        for traj_key, traj in traj_feature_map.items():
            trajs_hash.append(self.traj2grid_seq(traj, isCoordinate))
        return trajs_hash

    def preprocess(self, traj_feature_map, isCoordinate = False):
        if not isCoordinate:
            # havent inspected code in this block
            traj_grids = self._traj2grid_preprocess(traj_feature_map, False) #  [ [ [grid_index] ] ] 
            print('gird trajectory nums {}'.format(len(traj_grids)))

            useful_grids = {}
            count = 0
            max_len = 0
            for i, traj in enumerate(traj_grids):
                if len(traj) > max_len: max_len = len(traj)
                count += len(traj)
                for grid in traj:
                    if grid in useful_grids:
                        useful_grids[grid][1] += 1
                    else:
                        useful_grids[grid] = [len(useful_grids) + 1, 1]
            print(len(useful_grids.keys()))
            print(count, max_len)
            return traj_grids, useful_grids, max_len
        elif isCoordinate:
            traj_grids = self._traj2grid_preprocess(traj_feature_map, isCoordinate = isCoordinate)
            max_len = 0
            useful_grids = {}
            for i, traj in enumerate(traj_grids):
                if len(traj) > max_len: max_len = len(traj)
            return traj_grids, useful_grids, max_len


# what happen when the len of traj is more than maxlen??
def pad_sequence(trajs, maxlen = 100, pad_value = 0.0):
    # paddec_seqs = []
    # for traj in traj_grids:
    #     pad_r = np.zeros_like(traj[0])*pad_value # ones_like?
    #     while (len(traj) < maxlen):
    #         traj.append(pad_r)
    #     paddec_seqs.append(traj)
    # return paddec_seqs
    hidden_dim = len(trajs[0][0]) # 4
    pad = [pad_value] * hidden_dim
    rtn = []
    for traj in trajs:
        if len(traj) < maxlen:
            rtn.append(traj + [pad] * (maxlen - len(traj)))
        else:
            rtn.append(traj)
    return rtn


# previously name as trajectory_feature_generation
def neutraj_trajs_preprocess(trajs, lat_range, lon_range, grid_sidelen):
    # lst_trajs = [ [[lon, lat], [lon, lat], ...], ...] 
    
    traj_dic = dict(list(enumerate(trajs))) # {index: traj_coordinates list}; only valid trajs 

    # latitude and longtitude ranges are given
    # grids are predefined
    preprocessor = Preprocesser(delta = grid_sidelen, lat_range = lat_range, lon_range = lon_range)
    logging.debug('Number of grids(lon*lat)={}*{}'.format(len(preprocessor.x) - 1, len(preprocessor.y) - 1))
    
    # cPickle.dump(traj_index, open('./features/{}_traj_index'.format(fname),'w')) # {index: traj_coordinates}
    
    # trajs_coor = [ [ [x_coor, y_coor] ] ]; for each traj, if sequential points are in the grid, only the first one are left.
    trajs_coor, _, max_traj_len = preprocessor.preprocess(traj_dic, isCoordinate = True) 

    # cPickle.dump((trajs,[],max_traj_len), open('./features/{}_traj_coord'.format(fname), 'w'))

    trajs_gridxy = []
    min_x_grid, min_y_grid, max_x_grid, max_y_grid = 2^31, 2^31, 0, 0 # predefined hyperparameter
    for traj in trajs_coor:
        for p in traj:
            gx, gy, index = preprocessor.get_grid_index((p[0], p[1]))
            min_x_grid = gx if gx < min_x_grid else min_x_grid
            max_x_grid = gx if gx > max_x_grid else max_x_grid
            min_y_grid = gy if gy < min_y_grid else min_y_grid
            max_y_grid = gy if gy > max_y_grid else max_y_grid

    for traj in trajs_coor:
        traj_gridxy = []
        for p in traj:
            gx, gy, index = preprocessor.get_grid_index((p[0], p[1]))
            gx = gx - min_x_grid
            gy = gy - min_y_grid
            traj_gridxy.append([gx, gy])
        trajs_gridxy.append(traj_gridxy) # ${grid of traj} - ${grid offset}

    # _traj_index, never used?
    # dic = {'trajs_coor': trajs_coor, \
    #         'trajs_gridxy': trajs_gridxy, \
    #         'max_traj_len': max_traj_len}

    # cPickle.dump((trajs_gridxy, [], max_traj_len), open('./features/{}_traj_grid'.format(fname), 'w')) # [ [ [y_grid, x_grid] ] ]
    return trajs_coor, trajs_gridxy, max_traj_len


# another neutraj data preprocessing...., previously name as batch_generator
def neutraj_trajs_process_for_model_input(trajs_coor, trajs_gridxy, max_traj_len, x_range, y_range, \
                                            sam_spatial_width = 2):
            # griddatapath = Config.gridxypath,
            #         coordatapath = Config.corrdatapath,
            #         distancepath = Config.distancepath,
            #         train_radio = Config.seeds_radio):

    # dataset_length = Config.datalength
    # traj_grids, useful_grids, max_len = cPickle.load(open(griddatapath, 'r')) # a temp variable
    # self.trajs_length = [len(j) for j in traj_grids][:dataset_length]
    # self.grid_size = Config.gird_size # [1100, 1100]
    # self.max_length = max_len # max_traj_len
    trajs_length = list(map(len, trajs_coor))

    # add 2 to each grid traj, for SAM
    grid_trajs = [[[p[0]+sam_spatial_width, p[1]+sam_spatial_width] for p in traj] for traj in trajs_gridxy]


    # traj_coors, useful_grids, max_len = cPickle.load(open(coordatapath, 'r'))
    # x, y = [], [] # all x_coord values and y_coord values
    # for traj in traj_coors:
    #     for r in traj:
    #         x.append(r[0])
    #         y.append(r[1])

    # normalize coors - used previously in trajcl
    # xs, ys = zip(*[[p[0], p[1]] for traj in trajs_coor for p in traj])
    # meanx, meany, stdx, stdy = np.mean(xs), np.mean(ys), np.std(xs), np.std(ys)
    # coor_trajs = [[[(p[0] - meanx) / stdx, (p[1] - meany) / stdy] for p in traj] for traj in trajs_coor]
    
    # normalize  - used for effi
    x_min, x_max = x_range
    y_min, y_max = y_range
    coor_trajs = [[[(p[0] - x_min) / (x_max-x_min), (p[1] - y_min) / (y_max - y_min)] for p in traj] for traj in trajs_coor]
    

    # coordinators
    # coor_trajs = traj_coors[:dataset_length] # normalised coord trajs
    # train_size = int(len(grid_trajs)*train_radio/self.batch_size)*self.batch_size # training dataset size
    # print(train_size)
    
    # prepare train+test datasets
    # grid_train_seqs, grid_test_seqs = grid_trajs[:train_size], grid_trajs[train_size:] # exactly for training
    # coor_train_seqs, coor_test_seqs = coor_trajs[:train_size], coor_trajs[train_size:]

    # self.grid_trajs = grid_trajs
    # self.grid_train_seqs = grid_train_seqs
    # self.coor_trajs = coor_trajs
    # self.coor_train_seqs = coor_train_seqs
    
    # pad_trjs = [] # [ [normalised coord traj-x, normalised coord traj-y, grid-x, grad-y] ]
    # for i, t in enumerate(grid_trajs):
    #     traj = []
    #     for j, p in enumerate(t):
    #         # [-1.1607499926658438, 1.299966075551713, 34, 202]
    #         traj.append([coor_trajs[i][j][0], coor_trajs[i][j][1], p[0], p[1]]) 
    #     pad_trjs.append(traj)
    # print(pad_trjs[0])

    trajs_pad_input = [] # list of list of list; each point is 4-dim list
    for _traj_coor, _traj_grid in zip(*(coor_trajs, grid_trajs)):
        _lst = [ [_point_coor[0], _point_coor[1], _point_grid[0], _point_grid[1]] \
                    for _point_coor, _point_grid in zip(*(_traj_coor, _traj_grid))]
        trajs_pad_input.append(_lst)

    trajs_pad_input = pad_sequence(trajs_pad_input, max_traj_len)
    # print("Padded Trajs shape")
    # print(len(pad_trjs))
    # self.train_seqs = pad_trjs[:train_size]
    # self.padded_trajs = np.array(pad_sequence(pad_trjs, maxlen= max_len))

    # distance = cPickle.load(open(distancepath,'r'))
    # max_dis = distance.max()
    # print('max value in distance matrix :{}'.format(max_dis))
    # print(Config.distance_type)
    # if Config.distance_type == 'dtw':
    #     distance = distance/max_dis
    # print("Distance shape")
    # print(distance[:train_size].shape)
    # train_distance = distance[:train_size, :train_size]
    # print(distance[0])

    # print("Train Distance shape")
    # print(train_distance.shape)
    # self.distance = distance
    # self.train_distance = train_distance

    return trajs_pad_input, trajs_length


def distance_sampling(distance, train_seq_len, index, sampling_num, neutraj_mail_pre_degree=8):
    index_dis = distance[index]
    pre_sort = [np.exp(-i*neutraj_mail_pre_degree) for i in index_dis[:train_seq_len]]
    sample_index = []
    t = 0
    importance = []
    for i in pre_sort/np.sum(pre_sort):
        importance.append(t)
        t+=i
    importance = np.array(importance)
    # print(importance)
    while len(sample_index)<sampling_num:
        a = np.random.uniform()
        idx = np.where(importance>a)[0]
        if len(idx)==0: sample_index.append(train_seq_len-1)
        elif ((idx[0]-1) not in sample_index) & (not ((idx[0]-1) == index)): sample_index.append(idx[0]-1)
    sorted_sample_index = []
    for i in sample_index:
        sorted_sample_index.append((i, pre_sort[i]))
    sorted_sample_index = sorted(sorted_sample_index, key= lambda a:a[1], reverse=True)
    return [i[0] for i in sorted_sample_index]


def negative_distance_sampling(distance, train_seq_len, index, sampling_num, neutraj_mail_pre_degree=8):
    index_dis = distance[index]
    pre_sort = [np.exp(-i * neutraj_mail_pre_degree) for i in index_dis[:train_seq_len]]
    pre_sort = np.ones_like(np.array(pre_sort)) - pre_sort
    # print([(i,j) for i,j in enumerate(pre_sort)])
    sample_index = []
    t = 0
    importance = []
    for i in pre_sort / np.sum(pre_sort):
        importance.append(t)
        t += i
    importance = np.array(importance)
    # print(importance)
    while len(sample_index) < sampling_num:
        a = np.random.uniform()
        idx = np.where(importance > a)[0]
        if len(idx) == 0:
            sample_index.append(train_seq_len - 1)
        elif ((idx[0] - 1) not in sample_index) & (not ((idx[0] - 1) == index)):
            sample_index.append(idx[0] - 1)
    sorted_sample_index = []
    for i in sample_index:
        sorted_sample_index.append((i, pre_sort[i]))
    sorted_sample_index = sorted(sorted_sample_index, key=lambda a: a[1], reverse=True)
    return [i[0] for i in sorted_sample_index]


def trajcoor_to_trajpadinput(lst_trajs: list, lon_range, lat_range, cell_size):
    # lst_trajs = [[[lon, lat], []], ...] 


    # 2. init grid space (use diyao's code rather instead of our CellSpace)
    # 3. convert traj_xy -> traj_grid (use neutraj cellspace)
    # 4. normalize traj_xy and pad sequence

    logging.debug("trajcoor_to_trajcoor_and_trajgrid starts")
    _time = time.time()

    # 2. 3. 4.
    # x_range = [self.cs.x_min, self.cs.x_max]
    # y_range = [self.cs.y_min, self.cs.y_max]
    x_range, y_range = lon_range, lat_range
    trajs_coor, trajs_gridxy, max_traj_len = neutraj_trajs_preprocess(lst_trajs, y_range, x_range, cell_size)
    trajs_pad_input, trajs_length = neutraj_trajs_process_for_model_input( \
                                                        trajs_coor, trajs_gridxy, max_traj_len, \
                                                        x_range, y_range)

    logging.debug("trajcoor_to_trajpadinput ends. @={:.3f}, #trajs={}" \
                .format(time.time() - _time, len(trajs_pad_input)))
    #trajs_pad_input : list?
     # inputs_arrays = [[ [-1.1607499926658438, 1.299966075551713, 34.0, 202.0] ]]
    return trajs_pad_input, trajs_length
