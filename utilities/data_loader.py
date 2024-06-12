import os
import time
import math
import h5py
import random
import logging
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# DONT depend on config.py

# Use this function in order to read pd file in one time,
# then we split the dataset into 3 partitions.
# Load dataset for trajsimi learning.
def read_traj_dataset(file_path, num_trajs):
    logging.info('[Load traj dataset] START.')
    _time = time.time()
    trajs = pd.read_pickle(file_path)

    l = trajs.shape[0]
    if 'germany' in file_path: # TODO: hardcode while revision
        train_idx = (0, 30000)
        eval_idx = (30000, 40000)
        test_idx = (-100001, -1)
    else:
        train_idx = (int(l*0), num_trajs)
        eval_idx = (int(l*0.7), int(l*0.8))
        test_idx = (int(l*0.8), int(l*1.0))

    _train = TrajDataset(trajs[train_idx[0]: train_idx[1]])
    _eval = TrajDataset(trajs[eval_idx[0]: eval_idx[1]])
    _test = TrajDataset(trajs[test_idx[0]: test_idx[1]])

    logging.info('[Load traj dataset] END. @={:.0f}, #={}({}/{}/{})' \
                .format(time.time() - _time, l, len(_train), len(_eval), len(_test)))
    return _train, _eval, _test


class TrajDataset(Dataset):
    def __init__(self, data):
        self.data = data # DataFrame

    def __getitem__(self, index):
        return self.data.loc[index].merc_seq

    def __len__(self):
        return self.data.shape[0]


class DatasetST_h5(Dataset):
    # dataset for spatial (or spatio-temporal) trajectories
    def __init__(self, file_path: str, use_temporal: bool, num_trajs, traj_len_min, traj_len_max, seed):
        # use_temporal: True:spatio-temporal. False:spatial
        
        random.seed(seed)
        self.use_temporal = use_temporal
        self.traj_len_min = traj_len_min
        self.traj_len_max = traj_len_max
        self.data_length = num_trajs

        self.datafile = h5py.File(file_path, 'r')
        self.merc_range = self.datafile.attrs['merc_range'][()]
        self.ts_range = self.datafile.attrs['ts_range'][()] if 'ts_range' in self.datafile.attrs.keys() else None
        trajs_len = self.datafile['/trajs_len'][:].tolist()
        
        all_indices = list(filter(lambda iv: iv[1] >= traj_len_min, enumerate(trajs_len))) # list of tuple
        all_indices = list(map(self.__truncate_length, all_indices)) # list of tuple
        assert len(all_indices) >= math.sqrt(num_trajs)
        sampled_indices = list(zip(random.choices(all_indices, k = num_trajs), 
                                        random.choices(all_indices, k = num_trajs))) # list of tuple of tuple; replacement sample

        # must read everything here.
        _time = time.time()
        self.trajs = []
        for (idx1, length1), (idx2, length2) in sampled_indices:
            traj1 = self.datafile['/trajs_merc/%s' % idx1][ : length1].tolist()
            traj2 = self.datafile['/trajs_merc/%s' % idx2][ : length2].tolist()
            if self.use_temporal == False:
                self.trajs.append( (traj1, traj2) )
            else:
                traj1_ts = self.datafile['/trajs_ts/%s' % idx1][ : length1].tolist()
                traj1_xyts = [[p[0], p[1], traj1_ts[i]] for i, p in enumerate(traj1)]
                traj2_ts = self.datafile['/trajs_ts/%s' % idx2][ : length2].tolist()
                traj2_xyts = [[p[0], p[1], traj2_ts[i]] for i, p in enumerate(traj2)]
                self.trajs.append( (traj1_xyts, traj2_xyts) )
       
        all_indices.clear()
        sampled_indices.clear()
        trajs_len.clear()
        self.datafile.close()
        logging.info("[DatasetST_h5] file loaded. {}. st={}. @={:.4f}".format( \
                        file_path, use_temporal, time.time() - _time))
        
        
    def __getitem__(self, index):
        return self.trajs[index]


    def __len__(self):
        return self.data_length

    
    def __truncate_length(self, iv):
        # truncating trajectories longer than traj_len_max.
        # otherwise, no enough trajs...
        if iv[1] > self.traj_len_max:
            v = random.randint(self.traj_len_min, self.traj_len_max)
            return (iv[0], v)
        return iv
        

class DatasetSynthetic(Dataset):
    # dataset for spatial (or spatio-temporal) trajectories
    def __init__(self, use_temporal: bool, num_trajs, traj_len_min, traj_len_max, seed):
        # use_temporal: True:spatio-temporal. False:spatial
        
        random.seed(seed)
        self.use_temporal = use_temporal
        self.traj_len_min = traj_len_min
        self.traj_len_max = traj_len_max
        self.num_trajs = num_trajs
        self.merc_range = [0.0, 1.0, 0.0, 1.0]
        
    def __getitem__(self, index):
        
        traj_len = np.random.randint(self.traj_len_min, self.traj_len_max + 1)
        traj = np.random.uniform(size=(traj_len, 2)).astype(np.float64).tolist()
        
        traj2_len = np.random.randint(self.traj_len_min, self.traj_len_max + 1)
        traj2 = np.random.uniform(size=(traj2_len, 2)).astype(np.float64).tolist()
        if not self.use_temporal:
            return (traj, traj2)
        else:
            ts = np.cumsum(np.random.randint(1, 60, size = (traj_len)), axis = 0).tolist()
            ts2 = np.cumsum(np.random.randint(1, 60, size = (traj2_len)), axis = 0).tolist()

            traj = [[p[0], p[1], t] for p, t in zip(traj, ts)]
            traj2 = [[p[0], p[1], t] for p, t in zip(traj2, ts2)]
            return (traj, traj2)

    def __len__(self):
        return self.num_trajs
    

# Load traj dataset for trajsimi learning
def read_trajsimi_traj_dataset(file_path):
    logging.info('[Load trajsimi traj dataset] START.')
    _time = time.time()

    df_trajs = pd.read_pickle(file_path)
    if 'germany' in file_path:
        offset_idx = 30000 # use eval dataset
    else:
        offset_idx = int(df_trajs.shape[0] * 0.7) # use eval dataset
    df_trajs = df_trajs.iloc[offset_idx : offset_idx + 10000]
    assert df_trajs.shape[0] == 10000
    l = 10000

    train_idx = (int(l*0), int(l*0.7))
    eval_idx = (int(l*0.7), int(l*0.8))
    test_idx = (int(l*0.8), int(l*1.0))
    trains = df_trajs.iloc[train_idx[0] : train_idx[1]]
    evals = df_trajs.iloc[eval_idx[0] : eval_idx[1]]
    tests = df_trajs.iloc[test_idx[0] : test_idx[1]]

    logging.info("trajsimi traj dataset sizes. traj: #total={} (trains/evals/tests={}/{}/{})" \
                .format(l, trains.shape[0], evals.shape[0], tests.shape[0]))
    return trains, evals, tests


# Load simi dataset for trajsimi learning
def read_trajsimi_simi_dataset(file_path):
    logging.info('[Load trajsimi simi dataset] START.')
    _time = time.time()
    if not os.path.exists(file_path):
        logging.error('trajsimi simi dataset does not exist')
        exit(200)

    with open(file_path, 'rb') as fh:
        trains_simi, evals_simi, tests_simi, max_distance = pickle.load(fh)
        logging.info("[trajsimi simi dataset loaded] @={}, trains/evals/tests={}/{}/{}" \
                .format(time.time() - _time, len(trains_simi), len(evals_simi), len(tests_simi)))
        return trains_simi, evals_simi, tests_simi, max_distance

