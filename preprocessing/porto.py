import sys
sys.path.append('..')
import os
import math
import time
import logging
import pickle
import pandas as pd
from ast import literal_eval
import numpy as np
import traj_dist.distance as tdist
import multiprocessing as mp
from functools import partial

from config import Config
# from utils import tool_funcs
# from utils.cellspace import CellSpace
# from utils.tool_funcs import lonlat2meters
# from model.node2vec_ import train_node2vec
# from utils.traj_dist3 import edwp
# from utils.data_loader import read_trajsimi_traj_dataset, read_traj_dataset


# ref: TrjSR
def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))


def inrange(lon, lat):
    if lon <= Config.min_lon or lon >= Config.max_lon \
            or lat <= Config.min_lat or lat >= Config.max_lat:
        return False
    return True


def update_range(src_min, src_max, tgt_min, tgt_max):
    src_min = src_min if src_min < tgt_min else tgt_min
    src_max = src_max if src_max > tgt_max else tgt_max
    return src_min, src_max


def clean_and_output_data():
    _time = time.time()
    input_file = os.path.join(Config.data_dir, 'porto_raw_1.7m')
    df = pd.read_csv(input_file)
    print('Raw. #traj={}'.format(df.shape[0]), flush = True)
    
    df = df.rename(columns = {"POLYLINE": "wgs_seq"})
    df = df[df.MISSING_DATA == False]

    # length requirement
    df.wgs_seq = df.wgs_seq.apply(literal_eval)
    df['trajs_len'] = df.wgs_seq.apply(lambda traj: len(traj))
    df = df[(df.trajs_len >= Config.min_traj_len) & (df.trajs_len <= Config.max_traj_len)]
    print('Preprocessed-verification. #traj={}'.format(df.shape[0]), flush = True)
    
    # range requirement
    df['inrange'] = df.wgs_seq.map(lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj) ) # True: valid
    df = df[df.inrange == True]
    print('Preprocessed-verification. #traj={}'.format(df.shape[0]), flush = True)

    # convert to Mercator
    df['merc_seq'] = df.wgs_seq.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])
    df['x_min_merc'] = df.merc_seq.map(lambda traj: min([p[0] for p in traj]) )
    df['x_max_merc'] = df.merc_seq.map(lambda traj: max([p[0] for p in traj]) )
    df['y_min_merc'] = df.merc_seq.map(lambda traj: min([p[1] for p in traj]) )
    df['y_max_merc'] = df.merc_seq.map(lambda traj: max([p[1] for p in traj]) )
    x_min_merc, x_max_merc, y_min_merc, y_max_merc = \
            df['x_min_merc'].min(), df['x_max_merc'].max(), df['y_min_merc'].min(), df['y_max_merc'].max()
    print('merc range: {}, {}, {}, {}'.format(x_min_merc, x_max_merc, y_min_merc, y_max_merc), flush = True)
    print('Preprocessed-outputting. @={:.0f}, #traj={}'.format(time.time() - _time, df.shape[0]), flush = True)

    trajs_merc = df['merc_seq'].to_list()
    trajs_len = df['trajs_len'].to_list()
    dataset = {'n':len(trajs_len), 'trajs_merc':trajs_merc, \
                    'trajs_ts':None, 'trajs_len':trajs_len, \
                    'merc_range': (x_min_merc, x_max_merc, y_min_merc, y_max_merc), 'ts_range': None}
    
    with open(Config.dataset_file, 'wb') as fh:
        pickle.dump(dataset, fh, protocol = pickle.HIGHEST_PROTOCOL)
    
    print('Preprocess end. @={:.0f}'.format(time.time() - _time), flush = True)
    return


# ===========Do set dataset_file name in config.py!!===========
# cd utils
# nohup python porto.py &> ../result &
if __name__ == '__main__':
    Config.dataset = 'porto'
    Config.post_value_updates()

    clean_and_output_data()
