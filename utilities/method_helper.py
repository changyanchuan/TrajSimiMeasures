import sys
sys.path.append('..')
from functools import partial
import numpy as np
import torch
import traj_dist.distance as tdist

from config import Config
from utilities import tool_funcs
from nn.utils.cellspace import CellSpace
import core
import core_cpu
from nn.T3S import T3S
from nn.TrjSR import TrjSR
from nn.TrajGAT_core import GraphTransformer as TrajGAT
from nn.TMN import TMN
from nn.RSTS import Encoder as RSTS
from nn.RSTS_utils import Region, load_rsts_region
from nn.TrajCL import TrajCL
from nn.NEUTRAJ_core import NeuTraj_Network as NEUTRAJ
from nn.MLP import MLP 


__all__ = ['is_heuristic', 'heuristic_fn_wrapper', 'learned_class_wrapper', 'heuristic_colllate_fn', 'learned_collate_fn_wrapper']


def is_heuristic(fn_name):
    heurs = ['dtw', 'erp', 'frechet', 'hausdorff', 'edr', 'lcss', 'sar', 'cdds'] 
    return any(list(map(lambda x: x in fn_name, heurs)))


def heuristic_fn_wrapper(fn_name):
    fn =  { 'dtw': core.dtw,
            'edr': core.edr,
            'erp': core.erp,
            'frechet': core.dfrechet,
            'hausdorff': core.hausdorff,
            'lcss': core.lcss,
            
            'cdds': core.cdds,
            'stedr': core.stedr,
            'stlcss': core.stlcss,
            'sar': core.sar,
            
            'dtw_cpu': tdist.dtw,
            'edr_cpu': tdist.edr,
            'erp_cpu': tdist.erp,
            'frechet_cpu': tdist.discret_frechet,
            'hausdorff_cpu': tdist.hausdorff,
            'lcss_cpu': tdist.lcss,
            
            'cdds_cpu': core_cpu.cdds,
            'stedr_cpu': core_cpu.stedr,
            'stlcss_cpu': core_cpu.stlcss,
            'sar_cpu': core_cpu.sar,
            }.get(fn_name, None)

    if fn_name in ['edr', 'lcss', 'edr_cpu', 'lcss_cpu', 'cdds', 'cdds_cpu']:
        fn = partial(fn, eps = Config.trajsimi_edr_lcss_eps)
    elif fn_name in ['stedr', 'stlcss', 'stedr_cpu', 'stlcss_cpu']:
        fn = partial(fn, eps = Config.trajsimi_edr_lcss_eps, delta = Config.trajsimi_edr_lcss_delta)
    elif fn_name in ['erp_cpu']:
        fn = partial(fn, g = np.asarray([0, 0], dtype = np.float64))
    elif fn_name in ['sar', 'sar_cpu']:
        fn = partial(fn, distance_eps = Config.trajsimi_sar_distance_eps, 
                     time_eps = Config.trajsimi_sar_time_eps, 
                     target_length = Config.trajsimi_sar_target_length)
    
    if fn_name in ['dtw', 'erp', 'frechet', 'hausdorff', 'edr', 'lcss', 'cdds', 'stedr' ,'stlcss']:
        fn = partial(fn, threads_per_traj = Config.effi_gpu_threads_per_traj)
        
    assert fn != None
    return fn


def learned_class_wrapper(name, region_range):
    device = torch.device("cuda:0") if Config.gpu else torch.device("cpu")
        
    lon_range = (region_range[0], region_range[1])
    lat_range = (region_range[2], region_range[3])

    method = None
    if name == 'T3S':
        cellspace = CellSpace(Config.cell_size, Config.cell_size, 
                                region_range[0], region_range[2], 
                                region_range[1], region_range[3])
        method = T3S(Config.cell_embedding_dim, Config.traj_embedding_dim, 
                        cellspace)
   
    elif name == 'TrjSR':
        method = TrjSR(lon_range, lat_range, Config.trjsr_imgsize_x_lr, 
                        Config.trjsr_imgsize_y_lr, Config.trjsr_pixelrange_lr,
                        Config.traj_embedding_dim)
    
    elif name == 'TMN':
        cellspace = CellSpace(Config.cell_size, Config.cell_size, 
                                region_range[0], region_range[2], 
                                region_range[1], region_range[3])
        method = TMN(4, Config.traj_embedding_dim, (cellspace.x_size, cellspace.y_size), 
                    Config.tmn_sampling_num, lon_range, lat_range, Config.cell_size)
   
    elif name == 'TrajGAT':
        method = TrajGAT(4, Config.traj_embedding_dim, Config.trajgat_num_head , Config.trajgat_num_encoder_layers, 
                            Config.trajgat_d_lap_pos, Config.trajgat_encoder_dropout, 
                            None, None, None, lon_range, lat_range)
        cp_file = '{}/exp/snapshot/{}_TrajGAT_{}'.format(Config.root_dir, Config.dataset_prefix, Config.traj_embedding_dim)
                                                        # Config.trajgat_qtree_node_capacity)

        method.load_checkpoint(cp_file, device) # aim to load qtree and its embeddings of nodes
    
    elif name == 'RSTS':
        region_file = '{}/exp/snapshot/{}_RSTS_region.pkl'.format(Config.root_dir, Config.dataset_prefix)
        rsts_region = load_rsts_region(region_file)
        method = RSTS(Config.cell_embedding_dim, Config.traj_embedding_dim, rsts_region.vocal_nums, 
                        Config.rsts_num_layers, Config.rsts_dropout, Config.rsts_bidirectional, 
                        rsts_region)
    elif name == 'TrajCL':
        cellspace = CellSpace(Config.cell_size, Config.cell_size, 
                                region_range[0], region_range[2], 
                                region_range[1], region_range[3])
        method = TrajCL(cellspace, Config.cell_embedding_dim)
    
    elif name == 'NEUTRAJ':
        cellspace = CellSpace(Config.cell_size, Config.cell_size, 
                                region_range[0], region_range[2], 
                                region_range[1], region_range[3])

        method = NEUTRAJ(4, Config.cell_embedding_dim, [cellspace.x_size, cellspace.y_size], 
                        Config.cell_size, lon_range, lat_range)
    
    elif name == 'MLP1':
        method = MLP(2, Config.cell_embedding_dim, nlayer = 1)
        
    elif name == 'MLP2':
        method = MLP(2, Config.cell_embedding_dim, nlayer = 2)

    assert method != None
    method.eval()
    return method.to(device)


def heuristic_colllate_fn(batch, is_pad):
    # is_pad: True -> gpu; False -> cpu
    lst_trajs, lst_trajs2 = zip(*batch)
    trajs_len = list(map(len, lst_trajs))
    trajs_len2 = list(map(len, lst_trajs2))
    if is_pad:
        lst_trajs = tool_funcs.pad_lists_to_array(lst_trajs + lst_trajs2)
        lst_trajs2 = np.copy(lst_trajs[-len(trajs_len2) : ])
        lst_trajs = np.copy(lst_trajs[ : len(trajs_len)])
        return lst_trajs, trajs_len, lst_trajs2, trajs_len2
    else: 
        return lst_trajs, trajs_len, lst_trajs2, trajs_len2


def learned_collate_fn_wrapper(name, model, batch = True):
    fn = None
    if name == 'T3S':
        if batch:
            from nn.T3S import collate_fn as T3S_collate_fn
        else:
            from nn.T3S import collate_fn_single as T3S_collate_fn
        fn = partial(T3S_collate_fn, cellspace = model.cellspace)
        
    elif name == 'TrjSR':
        if batch:
            from nn.TrjSR import collate_fn as TrjSR_collate_fn
        else:
            from nn.TrjSR import collate_fn_single as TrjSR_collate_fn
        fn = partial(TrjSR_collate_fn, lon_range = model.lon_range, 
                    lat_range = model.lat_range, imgsize_x_lr = model.imgsize_x_lr, 
                    imgsize_y_lr = model.imgsize_y_lr, pixelrange_lr = model.pixelrange_lr)
        
    elif name == 'TMN':
        if batch:
            from nn.TMN import collate_fn as TMN_collate_fn
        else:
            from nn.TMN import collate_fn_single as TMN_collate_fn
        fn = partial(TMN_collate_fn, lon_range = model.lon_range, 
                    lat_range = model.lat_range, cell_size = model.cell_size)
   
    elif name == 'TrajGAT':
        if batch:
            from nn.TrajGAT_core import collate_fn as TrajGAT_collate_fn
        else:
            from nn.TrajGAT_core import collate_fn_single as TrajGAT_collate_fn
        fn = partial(TrajGAT_collate_fn, qtree = model.qtree, 
                    qtree_name2id = model.qtree_name2id, 
                    x_range = model.x_range, y_range = model.y_range)
    
    elif name == 'RSTS':
        if batch:
            from nn.RSTS import collate_fn as RSTS_collate_fn
        else:
            from nn.RSTS import collate_fn_single as RSTS_collate_fn
        fn = partial(RSTS_collate_fn, region = model.region)
        
    elif name == 'TrajCL':
        if batch:
            from nn.TrajCL import collate_fn as TrajCL_collate_fn
        else:
            from nn.TrajCL import collate_fn_single as TrajCL_collate_fn
        fn = partial(TrajCL_collate_fn, cellspace = model.cellspace, embs = model.embs.data)
    
    elif name == 'NEUTRAJ':
        if batch:
            from nn.NEUTRAJ_core import collate_fn as NEUTRAJ_collate_fn
        else:
            from nn.NEUTRAJ_core import collate_fn_single as NEUTRAJ_collate_fn
        fn = partial(NEUTRAJ_collate_fn, x_range = model.x_range, y_range = model.y_range, cell_size = model.cell_size)
    
    elif name == 'MLP1' or name == 'MLP2':
        if batch:
            from nn.MLP import collate_fn as MLP_collate_fn 
        else:
            from nn.MLP import collate_fn_single as MLP_collate_fn 
        fn = MLP_collate_fn

    assert fn != None
    return fn


