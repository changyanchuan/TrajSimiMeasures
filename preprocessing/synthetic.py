import sys
sys.path.append('..')
import time
import logging
import pickle
from torch.utils.data.dataloader import DataLoader

from config import Config
from utilities.data_loader import DatasetSynthetic


def update_range(src_min, src_max, tgt_min, tgt_max):
    src_min = src_min if src_min < tgt_min else tgt_min
    src_max = src_max if src_max > tgt_max else tgt_max
    return src_min, src_max


def create_dataset(num_trajs, traj_len_min, traj_len_max):
    starting_time = time.time()
    trajs_merc = []
    trajs_ts = []
    trajs_len = []
    t_min, t_max = float('inf'), float('-inf')
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    
    dataset = DatasetSynthetic(True, num_trajs, traj_len_min, traj_len_max, 2000)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, drop_last = False,
                            collate_fn = lambda _x: _x)

    for batch in dataloader:
        traj3d, _ = batch[0]
        traj = []
        ts = []
        for p in traj3d:
            traj.append([p[0], p[1]])
            ts.append(p[2])
        trajs_merc.append(traj)
        trajs_ts.append(ts)
        trajs_len.append(len(traj))
        x_merc, y_merc = zip(*traj)
        x_min, x_max = update_range(x_min, x_max, min(x_merc), max(x_merc))
        y_min, y_max = update_range(y_min, y_max, min(y_merc), max(y_merc))
        t_min, t_max = update_range(t_min, t_max, min(ts), max(ts))

    assert len(trajs_merc) == num_trajs
    
    dataset = {'n':len(trajs_len), 'trajs_merc':list(trajs_merc), \
                    'trajs_ts':list(trajs_ts), 'trajs_len':list(trajs_len), \
                    'merc_range': (x_min, x_max, y_min, y_max), 'ts_range': (t_min, t_max)}
        
    logging.info('#trajs={}'.format(dataset['n']))
    logging.info('#merc_range={}'.format(dataset['merc_range']))
    logging.info('#ts_range={}'.format(dataset['ts_range']))
    
    with open(Config.dataset_file, 'wb') as fh:
        pickle.dump(dataset, fh, protocol = pickle.HIGHEST_PROTOCOL)

    logging.info('done. @={:.3f}'.format(time.time() - starting_time))
    return


# ===========Do set dataset_file name in config.py!!===========
# cd utils
# nohup python synthetic.py &> ../result &
if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG,
            format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
            handlers = [logging.StreamHandler()]
    )
    
    Config.dataset = 'synthetic'
    Config.post_value_updates()
    
    create_dataset(10000, 20, 200)

