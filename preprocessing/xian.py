import sys
sys.path.append('..')
import os
import math
import time
import logging
import pickle
import multiprocessing as mp

from config import Config


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


def traj_file_process_operator(file_name, chunk_start, chunk_end):
    # https://nurdabolatov.com/parallel-processing-large-file-in-python
    logging.info('fn start. {} - {}'.format(chunk_start, chunk_end))
    pid = os.getpid()
    with open(file_name, 'r') as f:
        f.seek(chunk_start)
        
        trajs = []
        trajs_ts = []
        trajs_merc = []
        trajs_len = []
        t_min, t_max = float('inf'), float('-inf')
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')

        for line in f:
            chunk_start += len(line)
            if chunk_start > chunk_end:
                break
            
            items = line.split(' ')
            items = items[3:]
            
            ts = list(map( float, items[0::3] ))
            x = list(map( float, items[1::3] ))
            y = list(map( float, items[2::3] ))
            traj = list(zip(x, y))
            
            if Config.min_traj_len <= len(traj) <= Config.max_traj_len:
                # trajs.append(traj)
                traj_merc = [lonlat2meters(p[0], p[1]) for p in traj]
                x_merc, y_merc = zip(*traj_merc)
                trajs_ts.append(ts)
                trajs_merc.append(traj_merc)
                trajs_len.append(len(traj))
                t_min, t_max = update_range(t_min, t_max, min(ts), max(ts))
                x_min, x_max = update_range(x_min, x_max, min(x_merc), max(x_merc))
                y_min, y_max = update_range(y_min, y_max, min(y_merc), max(y_merc))

    logging.info('fn end. {}. #traj={}'.format(chunk_end, len(trajs_len)))
    return [trajs_merc, trajs_ts, trajs_len, t_min, t_max, x_min, x_max, y_min, y_max]
      

def clean_and_output_data():
    starting_time = time.time()
    cpu_count = mp.cpu_count() - 5
    input_file = os.path.join(Config.data_dir, 'xian_201810_7_wgs')
    file_size = os.path.getsize(input_file)
    chunk_size = file_size // cpu_count

    chunk_args = []
    with open(input_file, 'r') as f:
        def is_start_of_line(position):
            if position == 0:
                return True
            f.seek(position - 1)
            return f.read(1) == '\n'

        def get_next_line_position(position):
            f.seek(position)
            f.readline()
            return f.tell()

        chunk_start = 0
        while chunk_start < file_size:
            chunk_end = min(file_size, chunk_start + chunk_size)

            while not is_start_of_line(chunk_end):
                chunk_end -= 1

            if chunk_start == chunk_end:
                chunk_end = get_next_line_position(chunk_end)

            args = (input_file, chunk_start, chunk_end)
            chunk_args.append(args)

            chunk_start = chunk_end
    
    logging.info('num of chunks={}'.format(len(chunk_args)))
    p = mp.Pool(len(chunk_args)) # TODO
    chunk_results = p.starmap(traj_file_process_operator, chunk_args)
    p.close()

    logging.info('multiprocessing done. @={:.3f}'.format(time.time() - starting_time))
    trajs_merc, trajs_ts, trajs_len, t_min, t_max, x_min, x_max, y_min, y_max = zip(*chunk_results)
    # trajs = sum(trajs, [])
    trajs_merc = sum(trajs_merc, [])
    trajs_ts = sum(trajs_ts, [])
    trajs_len = sum(trajs_len, [])
    t_min, t_max = min(t_min), max(t_max)
    x_min, x_max = min(x_min), max(x_max)
    y_min, y_max = min(y_min), max(y_max)
    
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
# nohup python xian.py &> ../result &
if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG,
            format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
            handlers = [logging.StreamHandler()]
    )
    
    Config.dataset = 'xian'
    Config.post_value_updates()
    
    clean_and_output_data()

