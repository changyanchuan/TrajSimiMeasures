import pandas as pd
from os import listdir
from os.path import isfile, join
import subprocess
import sys
import math

log_folder = "./slurm_log/tmp/"
output_csvfile = "./slurm_log/tmp/exp_result_tmp"


def read_log(filename, str_flag = 'EXPFlag'):
    cmd = "awk '/^(.*{}).*$/' {}".format(str_flag, filename)
    rtn = subprocess.check_output(cmd, shell = True)
    rtn = rtn.decode("utf-8").split('\n') # list of string

    lines = []
    for line in rtn:
        idx = line.find(str_flag)
        if idx == -1:
            continue
        lines.append(line[idx + len(str_flag) + 1 : ])
    return lines


def csv_split(s, separator):
    i = 1
    for v in s.split(separator):
        try:
            v = float(v)
        except ValueError:
            pass
        yield (str(i), v)
        i += 1


def str_kvs_split(s):
    for sub in s.split(","):
        k, _, v = sub.partition("=")
        try:
            v = float(v)
        except ValueError:
            pass
        yield (k, v)


def raw_logs_to_df(log_files, is_merge_df):
    if log_files == []:
        log_files = [f for f in listdir(log_folder) if isfile(join(log_folder, f)) and not f.startswith('.') ]
        log_files.sort() # os.listdir return file name list in arbitary order!
    print(log_files)

    if is_merge_df:
        df = pd.DataFrame()
    else:
        lst = []

    for f in log_files:
        str_results = read_log(log_folder + f)
        if not is_merge_df:
            lst.append(pd.DataFrame())

        for line in str_results:
            dic_kv = dict(str_kvs_split(line))
            # dic_kv = dict(csv_split(line, ' '))
            if is_merge_df:
                df = df.append(dic_kv, ignore_index=True)
            else:
                lst[-1] = lst[-1].append(dic_kv, ignore_index=True)

    return df if is_merge_df else lst


def output_df_to_csvfile(df, csvfile, exp_category):
    if exp_category in ['effi', 'knn']: # type(df) == dataframe

        if exp_category == 'effi':
            groupby_key = ['exp', 'fn', 'num_trajs', 'max_traj_len', 'gpu', 'dataset']
            # groupby_key = ['exp', 'fn', 'num_trajs', 'max_traj_len', 'gpu', 'dataset', 'traj_embedding_dim']
            sort_by_3 = 'num_trajs'
            sort_kv_3 = lambda x: int( math.log10(int(x)) * 1000 ) # 100 -> 2000, 1M ->6000

        elif exp_category == 'knn':
            groupby_key = ['exp', 'fn', 'dbset_size', 'max_traj_len', 'gpu', 'dataset']
            sort_by_3 = 'dbset_size'
            sort_kv_3 = lambda x: int( math.log10(int(x)) * 10000 ) # 100 -> 20000, 1M ->60000

        sort_by_1 = 'exp'
        sort_kv_1 = lambda x: {'effi_numtrajs': 10000000, 'effi_numpoints': 20000000, 'effi_embdim': 30000000, \
                                'knn_dbset_sizes': 40000000, 'knn_qset_numpoints': 50000000}[x]
        sort_by_5 = 'gpu'
        sort_kv_5 = lambda x: {'True': 1000000, '1.0': 1000000, 'False': 2000000, '0.0': 2000000,}[str(x)]
        sort_by_6 = 'dataset'
        sort_kv_6 = lambda x: {'porto': 100000, 'geolife': 200000, 'xian': 300000, 'synthetic': 400000, 'chengdu': 500000, 'germany': 600000}[x]
        sort_by_4 = 'max_traj_len'
        sort_kv_4 = lambda x: int( math.log2(int(x)/100) * 1000 ) # 200 -> 1000, 16000 -> 4000, 12800 -> 7000
        # sort_by_7 = 'traj_embedding_dim'
        # sort_kv_7 = lambda x: int( math.log2(int(x)) * 100 ) # 32 -> 500, 256 -> 800

        sort_by_2 = 'fn'
        sort_kv_2 = lambda x: {'dtw': 1, 'erp': 2, 'frechet': 3, 'hausdorff': 4, \
                                'DTW': 1, 'ERP': 2, 'Frechet': 3, 'Hausdorff': 4, \
                                'NEUTRAJ': 5, 'TMN': 6, 'T3S': 7, 'TrajCL': 8, \
                                'TrjSR': 9, 'TrajGAT': 10, \
                                'MLP1': 12, 'MLP2': 13, \
                                'stedr': 20, 'cdds': 21, 'sar': 22, 'RSTS': 23 }[x] 
                                    
        df = df.groupby(groupby_key).mean()
        df = df.reset_index()
        df = df.loc[ (df[sort_by_1].apply(sort_kv_1) + df[sort_by_2].apply(sort_kv_2) \
                    + df[sort_by_3].apply(sort_kv_3) + df[sort_by_4].apply(sort_kv_4) \
                    + df[sort_by_5].apply(sort_kv_5) + df[sort_by_6].apply(sort_kv_6) \
                    ).sort_values().index]
                    # + df[sort_by_7].apply(sort_kv_7)    ).sort_values().index]
        df.to_csv(csvfile, float_format='%.3f', index = False)
        
    return

'''
python log_parser.py effi
'''


if __name__ == '__main__':
    if len(sys.argv) == 1:
        exit -255

    exp_category = sys.argv[1]

    if len(sys.argv) == 3:
        log_files = [sys.argv[2]]
    else:
        log_files = []

    if exp_category in ['effi', 'knn']:
        df = raw_logs_to_df(log_files, True)
        output_df_to_csvfile(df, output_csvfile, exp_category)

    else:
        exit -254