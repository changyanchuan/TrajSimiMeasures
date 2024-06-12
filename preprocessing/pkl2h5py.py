import sys
sys.path.append('..')
import logging
import pickle
import h5py
import numpy as np

from config import Config


def pkl_to_h5py(pkl_file, h5py_file) :
    with open(pkl_file, 'rb') as fhr:
        dic_dataset = pickle.load(fhr)
        logging.info('pkl loaded.')

        with h5py.File(h5py_file, 'w') as fhw:
            fhw.attrs['n'] = dic_dataset['n']
            fhw.attrs['merc_range'] = dic_dataset['merc_range']
            if dic_dataset['ts_range']:
                fhw.attrs['ts_range'] = dic_dataset['ts_range']

            fhw.create_dataset('/trajs_len', data = dic_dataset['trajs_len'])
            for i in range(dic_dataset['n']):
                fhw.create_dataset('/trajs_merc/%s' % i, data = dic_dataset['trajs_merc'][i])
                # fhw.create_dataset('/trajs_len/%s' % i, data = dic_dataset['trajs_len'][i])
                if dic_dataset['trajs_ts']:
                    fhw.create_dataset('/trajs_ts/%s' % i, data = dic_dataset['trajs_ts'][i])
    logging.info('done.')

# ===========Do set dataset_file name in config.py!!===========
# cd utils
# nohup python pkl2h5py.py &> ../result &
if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG,
            format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
            handlers = [logging.StreamHandler()]
    )
    
    Config.dataset = 'geolife'
    Config.post_value_updates()
    
    pkl_file = Config.dataset_file
    h5py_file = Config.dataset_file + '.h5'
    
    pkl_to_h5py(pkl_file, h5py_file)
    
    