
import sys
sys.path.append('..')
import os
import math
import time
import random
import logging
import pickle
import numpy as np
import traj_dist.distance as tdist
import multiprocessing as mp
from functools import partial

from config import Config
import core_cpu

'''
To create datasets used for the experiments of trajsimi effectivness.
Example:
nohup python trajsimi.py &> ../result &
'''


def trajsimi_dataset_traj_creation():
    starting_time = time.time()
    _output_file = '{}_trajsimi_{}{}_dict_traj'.format(Config.dataset_file,
                                                       Config.trajsimi_min_traj_len,
                                                       Config.trajsimi_max_traj_len)

    # if traj file exists, read it from file directly
    if os.path.isfile(_output_file):
        with open(_output_file, 'rb') as fh:
            logging.info('traj file exists')
            dic_dataset = pickle.load(fh)
            trajs_merc = dic_dataset["trajs_merc"]
            trajs_ts = dic_dataset["trajs_ts"]
            
            if trajs_ts == None:
                return (trajs_merc[:7000], trajs_merc[7000:8000], trajs_merc[8000:]), None
            else:
                return (trajs_merc[:7000], trajs_merc[7000:8000], trajs_merc[8000:]), \
                        (trajs_ts[:7000], trajs_ts[7000:8000], trajs_ts[8000:])

    
    with open(Config.dataset_file, 'rb') as fh:
        dic_dataset = pickle.load(fh)
        logging.info("traj_simi_computation pkl loaded. @={:.3f}".format(time.time() - starting_time))
        
        trajs_len = dic_dataset['trajs_len']
        traj_len_min = Config.trajsimi_min_traj_len
        traj_len_max = Config.trajsimi_max_traj_len
        
        # selected trajectories satisfying len \in [20, 200]
        # allow truncation, otherwise, Geolife does not have enough datasets
        random.seed(2000)
        all_indices = list(filter(lambda iv: iv[1] >= traj_len_min, enumerate(trajs_len))) # list of tuple
        fn_truncate_length = partial(_truncate_length, traj_len_min = traj_len_min, traj_len_max = traj_len_max)
        all_indices = list(map(fn_truncate_length, all_indices)) # list of tuple

        if traj_len_min >= 800 and Config.dataset == 'xian':
            # TODO: remove this part of code. 
            # we add this to solve the issue - xian dont have 10000 trajectories of which length >= 801
            # there are only 6138 trajectories of which length >= 801.
            # this part code is dedicated for knn effectivess exp on #points of trajs (800, 1600]
            # Exp of one-off trajectory similarity computation dont need this code.
            all_indices = all_indices + all_indices
            sampled_indices = all_indices[-10000:]
            assert len(all_indices) >= 10000 , logging.info("all indices.len={}".format(len(all_indices)))
            all_indices.clear()
        
        else:
            assert len(all_indices) >= 10000 , logging.info("all indices.len={}".format(len(all_indices)))
            sampled_indices = list(random.choices(all_indices, k = 10000)) # list of tuple
            all_indices.clear()

        sampled_trajs_merc = [dic_dataset['trajs_merc'][idx][:length] for idx, length in sampled_indices]
        sampled_trajs_idx, sampled_trajs_len = zip(*sampled_indices)
        sampled_trajs_ts = [dic_dataset['trajs_ts'][idx][:length] for idx, length in sampled_indices] if dic_dataset['trajs_ts'] != None else None
        
        # dump to file
        output_dic = {'trajs_merc':sampled_trajs_merc, 'trajs_ts':sampled_trajs_ts,
                      'trajs_idx':sampled_trajs_idx, 'trajs_len':sampled_trajs_len, }
    
        with open(_output_file, 'wb') as fh:
            pickle.dump(output_dic, fh, protocol = pickle.HIGHEST_PROTOCOL)
            logging.info("traj dumpped. @={:.3f}".format(time.time() - starting_time))

        train_trajs_merc, eval_trajs_merc, test_trajs_merc = sampled_trajs_merc[ : 7000], sampled_trajs_merc[7000 : 8000], sampled_trajs_merc[8000 : ]

        if sampled_trajs_ts == None:
            return (train_trajs_merc, eval_trajs_merc, test_trajs_merc), None
        else:
            train_trajs_ts, eval_trajs_ts, test_trajs_ts = sampled_trajs_ts[ : 7000], sampled_trajs_ts[7000 : 8000], sampled_trajs_ts[8000 : ]
            return (train_trajs_merc, eval_trajs_merc, test_trajs_merc), \
                    (train_trajs_ts, eval_trajs_ts, test_trajs_ts)
        

def trajsimi_dataset_simis_creation(trajs, tss):
    # spatial measures
    # trajsimi_dataset_simis_creation_fn(trajs, ['dtw', 'erp', 'dfrechet', 'hausdorff'])
    trajsimi_dataset_simis_creation_fn(trajs, ['hausdorff'])
    if tss == None:
        return
    
    # spatio-temporal measures
    train_trajs_merc, eval_trajs_merc, test_trajs_merc = trajs
    train_trajs_ts, eval_trajs_ts, test_trajs_ts = tss
    # 3D traj
    train_trajs_merc = [[ [train_trajs_merc[i][j][0], train_trajs_merc[i][j][1], train_trajs_ts[i][j]] \
                        for j in range(len(train_trajs_merc[i])) ] for i in range(len(train_trajs_merc)) ]
    eval_trajs_merc = [[ [eval_trajs_merc[i][j][0], eval_trajs_merc[i][j][1], eval_trajs_ts[i][j]] \
                        for j in range(len(eval_trajs_merc[i])) ] for i in range(len(eval_trajs_merc)) ]
    test_trajs_merc = [[ [test_trajs_merc[i][j][0], test_trajs_merc[i][j][1], test_trajs_ts[i][j]] \
                        for j in range(len(test_trajs_merc[i])) ] for i in range(len(test_trajs_merc)) ]
    
    trajs = train_trajs_merc, eval_trajs_merc, test_trajs_merc
    trajsimi_dataset_simis_creation_fn(trajs, ['stedr', 'cdds'])

    return


def trajsimi_dataset_simis_creation_fn(trajs, lst_fn_names):
    train_trajs_merc, eval_trajs_merc, test_trajs_merc = trajs
        
    for fn_name in lst_fn_names:
        
        starting_time = time.time()
        # ground-truth similarity computation
        fn = _get_simi_fn(fn_name)
        
        eval_simis = _simi_matrix(fn, eval_trajs_merc)
        test_simis = _simi_matrix(fn, test_trajs_merc)
        train_simis = _simi_matrix(fn, train_trajs_merc) # [ [simi, simi, ... ], ... ], variable length

        max_distance = max( max(map(partial(max, default = float('-inf')), eval_simis)), \
                            max(map(partial(max, default = float('-inf')), test_simis)), \
                            max(map(partial(max, default = float('-inf')), train_simis)) )

        output_dic = {'train_simis':train_simis, 'eval_simis':eval_simis, 'test_simis':test_simis, 
                        'max_distance':max_distance}

        _output_file = '{}_trajsimi_{}{}_dict_{}'.format(Config.dataset_file, 
                                                         Config.trajsimi_min_traj_len,
                                                         Config.trajsimi_max_traj_len,
                                                         fn_name)
        with open(_output_file, 'wb') as fh:
            pickle.dump(output_dic, fh, protocol = pickle.HIGHEST_PROTOCOL)
            logging.info("fn dumpped. fn={} @={:.3f}".format(fn_name, time.time() - starting_time))
    

def _truncate_length(iv, traj_len_min, traj_len_max):
    # iv: (index, length)
    # truncating trajectories longer than traj_len_max.
    # otherwise, no enough trajs...
    if iv[1] > traj_len_max:
        v = random.randint(traj_len_min, traj_len_max)
        return (iv[0], v)
    return iv


def _get_simi_fn(fn_name):
    fn =  {'lcss': tdist.lcss, 'edr': tdist.edr, 
            'erp': tdist.erp, 'dtw': tdist.dtw,
            'dfrechet': tdist.discret_frechet, 'hausdorff': tdist.hausdorff,
            'stedr': core_cpu.stedr, 'cdds': core_cpu.cdds,
            'stlcss': core_cpu.stlcss
            }.get(fn_name, None)
    if fn_name == 'erp': 
        fn = partial(fn, g = np.asarray([0, 0], dtype = np.float64))
        # fn = partial(fn, g = np.asarray([12125125, 4056355], dtype = np.float64)) # xian_7_20inf
    elif fn_name in ['stedr', 'stlcss']:
        fn = partial(fn, eps = Config.trajsimi_edr_lcss_eps, delta = Config.trajsimi_edr_lcss_delta)
    elif fn_name == 'cdds':
        fn = partial(fn, eps = Config.trajsimi_edr_lcss_eps)
    return fn


def _simi_matrix(fn, lst_trajs):
    
    _time = time.time()

    l = len(lst_trajs)
    batch_size = 50
    assert l % batch_size == 0

    # parallel init
    tasks = []
    for i in range(math.ceil(l / batch_size)):
        if i < math.ceil(l / batch_size) - 1:
            tasks.append( (fn, lst_trajs, list(range(batch_size * i, batch_size * (i+1)))) )
        else:
            tasks.append( (fn, lst_trajs, list(range(batch_size * i, l))) )
    
    num_cores = int(mp.cpu_count()) - 3
    logging.info("pool.size={}, task.size={}".format(num_cores, len(tasks)))
    pool = mp.Pool(num_cores)
    lst_simi = pool.starmap(_simi_comp_operator, tasks)
    pool.close()

    # extend lst_simi to matrix simi and pad 0s
    lst_simi = sum(lst_simi, [])
    for i, row_simi in enumerate(lst_simi):
        lst_simi[i] = [0]*(i+1) + row_simi
    arr_simi = np.asarray(lst_simi, dtype = np.float32)
    arr_simi = arr_simi + arr_simi.T
    assert arr_simi.shape[0] == arr_simi.shape[1] and arr_simi.shape[0] == l
        
    logging.info('simi_matrix computation done., @={}, #={}'.format(time.time() - _time, len(arr_simi)))

    return arr_simi # a square-shape np array


def _simi_comp_operator(fn, lst_trajs, sub_idx):
    # async operator
    # only compute the upper triangle, return list of list, variable lengths
    simi = []
    l = len(lst_trajs)
    for _i in sub_idx:
        t_i = np.asarray(lst_trajs[_i], dtype = np.float64)
        simi_row = []
        for _j in range(_i + 1, l):
            t_j = np.asarray(lst_trajs[_j], dtype = np.float64)
            simi_row.append(fn(t_i, t_j))
        simi.append(simi_row)
    logging.debug('simi_comp_operator ends. sub_idx=[{}:{}], pid={}' \
                    .format(sub_idx[0], sub_idx[-1], os.getpid()))
    return simi


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG,
            format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
            handlers = [logging.StreamHandler()]
    )
    
    Config.dataset = 'xian'
    Config.trajsimi_min_traj_len = 801 # default 20
    Config.trajsimi_max_traj_len = 1600 # default 200
    Config.post_value_updates()
    
    trajs, tss = trajsimi_dataset_traj_creation()
    trajsimi_dataset_simis_creation(trajs, tss)
