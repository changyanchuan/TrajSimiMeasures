import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append('./nn') # implicit calling - TrajGAT
import gc
import time
import logging
import argparse
import pickle
import pynvml
from functools import partial
from multiprocessing import Pool, Process
import shared_memory # py3.7 does not support multiprocessing.shared_memory
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from config import Config
from utilities import tool_funcs
from utilities.data_loader import DatasetSynthetic, DatasetST_h5
from utilities.method_helper import *
from nn.RSTS_utils import Region # implicit calling - RSTS


'''
nohup python test_trajsimi_time_DQ.py --dataset porto --effi_exp numtrajs --gpu &> result &
'''


def test_varying_num_trajs(datafile, seed):
    # varying number of trajectories in test sets
    
    num_trajs_lst = [(10**2, 10**3)] # dataset size of (Q and D)
    min_traj_len = Config.effi_min_traj_len
    max_traj_len = Config.effi_max_traj_len
    
    methods_s, stlabel_s = ['dtw', 'erp', 'frechet', 'hausdorff', 'T3S', 'TrjSR', 'TrajGAT', \
                            'TrajCL', 'NEUTRAJ', 'MLP2'], False
    methods_st, stlabel_st = ['stedr', 'cdds', 'sar', 'RSTS'], True
    
    if Config.effi_method != '':
        if Config.effi_method in methods_s:
            methods_s = [Config.effi_method]
            methods_st = []
        elif Config.effi_method in methods_st:
            methods_s = []
            methods_st = [Config.effi_method]
        else:
            methods_s = []
            methods_st = []
        
    for num_trajs_QD in num_trajs_lst:
        num_trajs = max(num_trajs_QD)
        for methods, stlabel in [(methods_s, stlabel_s), (methods_st, stlabel_st)]:
            if stlabel and Config.dataset in ['porto', 'germany']:
                continue
        
            dataset = DatasetST_h5(datafile, stlabel, num_trajs, min_traj_len, max_traj_len, seed)
            merc_range = dataset.merc_range

            for method_name in methods:
                
                shm_results = []
                base_ram, used_ram = tool_funcs.RAMInfo.mem_global(), 0
                base_gram, used_gram = tool_funcs.GPUInfo.mem()[0], 0
                logging.debug("Based memory usages RAM={}, GRAM={}".format(base_ram, base_gram))
                
                if Config.effi_auxilary_processor:
                    shm_results = shared_memory.ShareableList([0]*10)
                    aux_processor = Process(target = hardware_usage, args = (shm_results,))
                    aux_processor.start()
                    
                batch_size = Config.effi_batch_size_gpu if Config.gpu \
                            else (Config.effi_batch_size_cpu_heuristic if is_heuristic(method_name) else Config.effi_batch_size_cpu_learned)
                dl_num_workers = Config.effi_dataloader_num_workers
                
                metrics = trajsimi_computation_gpucpu(method_name, dataset, num_trajs_QD, \
                                                    merc_range, batch_size, Config.gpu, dl_num_workers)
                
                if Config.effi_auxilary_processor:
                    aux_processor.terminate()
                    used_ram = shm_results[1] - base_ram
                    used_gram = shm_results[5] - base_gram
                
                logging.info("[EXPFlag]exp=effi_numtrajs,dataset={},fn={},gpu={},"
                                "num_trajs={},min_traj_len={},max_traj_len={},"
                                "time={:.4f},coltime={:.4f},embtime={:.4f},comptime={:.4f},"
                                "ram={},gram={}".format( \
                                Config.dataset, method_name, Config.gpu, \
                                num_trajs, min_traj_len, max_traj_len, \
                                metrics[0], metrics[1], metrics[2], metrics[3], \
                                used_ram, used_gram))
                
                if Config.effi_auxilary_processor:
                    shm_results.shm.close()
                    shm_results.shm.unlink()
                    del shm_results
                    
                torch.cuda.empty_cache()
                gc.collect()
                
        torch.cuda.empty_cache()
        gc.collect()
            
    return 


def trajsimi_computation_gpucpu(method_name, dataset, num_trajs, region_range, batch_size, is_gpu, dataloader_num_workers = 0):
    num_trajs_q, num_trajs_d = num_trajs
    
    heuristic = is_heuristic(method_name)
    method_name = method_name + '_cpu' if (heuristic and not is_gpu) else method_name
    
    if heuristic:
        method_fn = heuristic_fn_wrapper(method_name)
        collate_fn = partial(heuristic_colllate_fn, is_pad = is_gpu) # TODO
    else:
        model = learned_class_wrapper(method_name, region_range) # on target device
        method_fn = model.interpret # on target device
        collate_fn = learned_collate_fn_wrapper(method_name, model, False)
        if not Config.gpu:
            torch.set_num_threads( min(Config.effi_batch_size_cpu_learned, Config.effi_cpu_method_num_cores))

    # D, Q = dataset[ : num_trajs_d], dataset[ : num_trajs_q]
    Q, D = zip(*dataset)
    Q = list(Q)[ : num_trajs_q]
    D = list(D)[ : num_trajs_d]
    Q_embs, D_embs = [], []
    
    collate_time = 0.0
    embedding_time = 0.0
    computation_time = 0.0
    running_time = 0.0
    starting_ts = time.time()
    
    if heuristic:
        _time = time.time()
        Q_len = list(map(len, Q))
        D_len = list(map(len, D))
        if is_gpu:
            Q_len = np.array(list(map(len, Q)))
            D_len = np.array(list(map(len, D)))
            Q = tool_funcs.pad_lists_to_array(Q + D)
            D = np.copy(Q[-num_trajs_d : ])
            Q = np.copy(Q[ : num_trajs_q])
        
        # idxs.shape = [2, num_trajs_q*num_trajs_d]
        idxs = np.stack((np.array(range(num_trajs_q)).repeat(num_trajs_d), \
                        np.tile(np.array(range(num_trajs_d)), num_trajs_q)  ))
        idxs_len = idxs.shape[1]
        collate_time += time.time() - _time
        
        start_idx = 0
        while start_idx < idxs_len:
            _time = time.time()

            end_idx = start_idx + batch_size if start_idx + batch_size < idxs_len else idxs_len
            if is_gpu:
                Q_batch = Q[idxs[0, start_idx:end_idx]]
                D_batch = D[idxs[1, start_idx:end_idx]]
                Q_len_batch = Q_len[idxs[0, start_idx:end_idx]]
                D_len_batch = D_len[idxs[1, start_idx:end_idx]]
            else:
                Q_batch = [Q[_i]  for _i in idxs[0, start_idx:end_idx] ]
                D_batch = [D[_i]  for _i in idxs[1, start_idx:end_idx] ]
                Q_len_batch = [Q_len[_i]  for _i in idxs[0, start_idx:end_idx]]
                D_len_batch = [D_len[_i]  for _i in idxs[1, start_idx:end_idx]]
                
            collate_time += time.time() - _time
            
            _time = time.time()
            if is_gpu:
                dists = method_fn(Q_batch, Q_len_batch, D_batch, D_len_batch)
                dists = dists.tolist()
            else:
                dists = trajsimi_computation_heuristic_cpu(method_fn, \
                                                Q_batch, Q_len_batch, D_batch, D_len_batch, \
                                                num_cores = Config.effi_cpu_method_num_cores)

            computation_time += time.time() - _time
            start_idx = end_idx
            
    else:
        try: # no timeout exception 
            start_idx = 0
            while start_idx < num_trajs_d:
                _time = time.time()
                
                end_idx = start_idx + batch_size if start_idx + batch_size < num_trajs_d else num_trajs_d
                batch = D[start_idx: end_idx]
                batch_collated = collate_fn(batch)
                collate_time += time.time() - _time
                
                _time = time.time()
                traj_embs = method_fn(batch_collated)
                D_embs.append(traj_embs)
                embedding_time += time.time() - _time
                
                start_idx = end_idx
            
            start_idx = 0
            while start_idx < num_trajs_q:
                _time = time.time()

                end_idx = start_idx + batch_size if start_idx + batch_size < num_trajs_q else num_trajs_q
                batch = Q[start_idx: end_idx]
                batch_collated = collate_fn(batch)
                collate_time += time.time() - _time
                
                _time = time.time()
                traj_embs = method_fn(batch_collated)
                Q_embs.append(traj_embs)
                embedding_time += time.time() - _time
                
                start_idx = end_idx
                
            D_embs = torch.cat(D_embs)
            Q_embs = torch.cat(Q_embs)
            _time = time.time()
            dists = torch.cdist(Q_embs, D_embs, p = 1)
            computation_time += time.time() - _time
            assert list(dists.size()) == [num_trajs_q, num_trajs_d]

        except Exception as excpt:
            logging.error("Failed. dataset={},fn={},gpu={},error={}--{}".format( \
                                    Config.dataset, method_name, Config.gpu, \
                                    type(excpt), excpt))
            return [-1, -1, -1, -1]

    if heuristic:
        pass
    else:
        del model
    running_time = time.time() - starting_ts
    
    return [running_time, collate_time, embedding_time, computation_time]


# multiprocessing-based heuristic trajsimi computation on CPU
def trajsimi_computation_heuristic_cpu(fn, lst_trajs, trajs_len, lst_trajs2, trajs_len2, num_cores):
    _time = time.time()    
    n = len(lst_trajs)
    
    if n == 1: # batch_size = 1: dont use multi-cores
        dists = _heuristic_cpu_operator(fn, lst_trajs, trajs_len, lst_trajs2, trajs_len2)
        assert len(dists) == n
        return dists
            
    slice_idx = tool_funcs.slicing(n, num_cores)
    tasks = []
    
    for start_idx, end_idx in slice_idx:
        tasks.append( (fn, lst_trajs[start_idx: end_idx], trajs_len[start_idx: end_idx], 
                            lst_trajs2[start_idx: end_idx], trajs_len2[start_idx: end_idx]) )
    logging.debug("t1 = {}".format(time.time() - _time))
    
    pool = Pool(num_cores)
    dists = pool.starmap(_heuristic_cpu_operator, tasks)
    pool.close()
    
    logging.debug("t2 = {}".format(time.time() - _time))
    dists = sum(dists, [])
    assert len(dists) == n
    
    return dists
    
    
def _heuristic_cpu_operator(fn, lst_trajs, trajs_len, lst_trajs2, trajs_len2):
    _time = time.time()
    dists = []
    for i in range(len(lst_trajs)):
        traj1 = np.asarray(lst_trajs[i][:trajs_len[i]])
        traj2 = np.asarray(lst_trajs2[i][:trajs_len2[i]])
        dists.append( fn(traj1, traj2) )
    logging.debug("t3 = {}, pid={}".format(time.time()-_time, os.getpid()))
    return dists


# function for the async processor to record hardware usage
def hardware_usage(results):
    # results: on shared memory
    pynvml.nvmlInit()
    _h = pynvml.nvmlDeviceGetHandleByIndex(0)
    n = 0
    while True:
        time.sleep(0.01)
        n += 1
        ram = tool_funcs.RAMInfo.mem_global()
        cpu = tool_funcs.CPUInfo.usage_percent()
        gpuram = tool_funcs.GPUInfo.mem(_h)[0]
        gpupower = tool_funcs.GPUInfo.power(_h)
        results[0] = (results[0]*(n-1)+ram)/n # avg ram usage
        results[1] = max(results[1], ram) # max ram usage
        results[2] = (results[2]*(n-1)+cpu)/n # avg cpu
        results[3] = max(results[3], cpu) # max cpu
        results[4] = (results[4]*(n-1)+gpuram)/n # avg gpu ram
        results[5] = max(results[5], gpuram) # max gpu ram
        results[6] = (results[6]*(n-1)+gpupower)/n # avg gpu power
        results[7] = max(results[7], gpupower) # max gpu power


def parse_args():
    parser = argparse.ArgumentParser(description = "...")
    # Font give default values here, because they will be faultly 
    # overwriten by the values in config.py.
    # config.py is the correct place for default values
    parser.add_argument('--dumpfile_uniqueid', type = str, help = '') # see config.py
    parser.add_argument('--seed', type = int, help = '')
    parser.add_argument('--dataset', type = str, help = '')
    parser.add_argument('--gpu', dest = 'gpu', action='store_true')
    parser.add_argument('--effi_exp', type = str, help = '')
    parser.add_argument('--effi_method', type = str, help = '')
    parser.add_argument('--effi_batch_size_cpu_heuristic', type = int, help = '')
    parser.add_argument('--effi_batch_size_cpu_learned', type = int, help = '')
    parser.add_argument('--effi_batch_size_gpu', type = int, help = '')
    
    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


def pre_compile(): # pre-compiling the cuda-based heuristic methods, in order to avoid jit
    def _fn(methods, use_temporal):
        dataset = DatasetSynthetic(use_temporal, 100, 20, 200, 2000)
        dataloader = DataLoader(dataset, batch_size = 100, shuffle=False, drop_last = False, \
                                collate_fn = partial(heuristic_colllate_fn, is_pad = True))

        for method_name in methods:
            for batch in dataloader:
                method_fn = heuristic_fn_wrapper(method_name)
                dists = method_fn(*batch)
        return 
    
    _fn(['dtw', 'erp', 'frechet', 'hausdorff'], False)
    _fn(['stedr', 'stlcss', 'sar', 'cdds'], True)
    logging.info('pre_jit done.')


if __name__ == '__main__':
    Config.update(parse_args())

    logging.basicConfig(level = logging.DEBUG if Config.debug else logging.INFO,
            format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
            handlers = [logging.FileHandler(Config.root_dir+'/exp/log/'+tool_funcs.log_file_name(), mode = 'w'), 
                        logging.StreamHandler()] )
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('dgl').setLevel(logging.ERROR)
    

    logging.info('python ' + ' '.join(sys.argv))
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    pre_compile()
    
    starting_time = time.time()
    
    datafile = Config.dataset_file + '.h5'
    if Config.effi_exp == 'numtrajs':
        test_varying_num_trajs(datafile, Config.seed)
    
    logging.info('all finished. @={:.1f}'.format(time.time() - starting_time))
    