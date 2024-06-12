import sys
sys.path.append('..')
import logging
import pickle
import numpy as np

from config import Config

def lonlat_statistics(trajs_lst):
    lonlats = []
    for t in trajs_lst:
        for p in t:
            lonlats.append(p)

    lonlats = np.asarray(lonlats)
    mean = np.mean(lonlats, axis = 0)
    std = np.std(lonlats, axis = 0)
    return mean[0], std[0], mean[1], std[1]


def create_trajsimi_data_file(traj_file, simi_file):
    
    with open(Config.dataset_trajsimi_traj, 'rb') as fh:
        dic_dataset = pickle.load(fh)
        trajs_merc = dic_dataset['trajs_merc']
        
        print("lon lat statistics:")
        print(lonlat_statistics(trajs_merc))
        
        with open(traj_file, "wb") as fh:
            pickle.dump(trajs_merc, fh, protocol = pickle.HIGHEST_PROTOCOL)
            logging.info("traj dump done.")
        
    with open(Config.dataset_trajsimi_dict, 'rb') as fh:
        dic_dataset = pickle.load(fh)
        train_simis = dic_dataset['train_simis']
        eval_simis = dic_dataset['eval_simis']
        test_simis = dic_dataset['test_simis']

        simi_arr = np.zeros([len(trajs_merc), len(trajs_merc)])
        simi_arr[:7000, :7000] = train_simis
        simi_arr[7000:8000, 7000:8000] = eval_simis
        simi_arr[8000:, 8000:] = test_simis

        with open(simi_file, "wb") as fh:
            pickle.dump(simi_arr, fh, protocol = pickle.HIGHEST_PROTOCOL)
            logging.info("simi dump done.")
    return

# nohup python ./preprocessing_dataset_TrajGAT.py &> ../result &
if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG,
                        format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers = [logging.StreamHandler()]
                        )
                        
    Config.dataset = 'xian'
    for fn_name in ['dtw', 'erp', 'dfrechet', 'hausdorff']:
        Config.trajsimi_measure = fn_name
        Config.trajsimi_min_traj_len = 201
        Config.trajsimi_max_traj_len = 400
        Config.post_value_updates()

        traj_file = Config.dataset_file + '_TrajGAT_trajsimi_{}{}_traj.pkl'.format(Config.trajsimi_min_traj_len, Config.trajsimi_max_traj_len)
        simi_file = Config.dataset_file + '_TrajGAT_trajsimi_{}{}_'.format(Config.trajsimi_min_traj_len, Config.trajsimi_max_traj_len) + fn_name + '_simi.pkl'
        
        create_trajsimi_data_file(traj_file, simi_file)
