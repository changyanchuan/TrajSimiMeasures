import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append('./nn') # implicit calling - TrajGAT
import faiss
import gc
import time
import logging
import argparse
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from config import Config
from utilities import tool_funcs
from utilities.data_loader import DatasetST_h5
from utilities.method_helper import *
from nn.RSTS_utils import Region # implicit calling - RSTS


'''
nohup python test_knn_time.py --dataset geolife --knn_exp dbset_sizes --gpu  &> result & 
'''

def test_varying_dbset_sizes():
    if Config.gpu:
        faiss.omp_set_num_threads(Config.knn_query_threads_gpu)
    else:  
        faiss.omp_set_num_threads(Config.knn_query_threads_cpu)

    qset_size = 1000
    dbset_trajlen_min = 20
    dbset_trajlen_max = 200
    qset_trajlen_min = 20
    qset_trajlen_max = 200
    
    dbset_sizes = [10**5] # default size
    # dbset_sizes = [10**4, 10**5, 10**6]
    
    methods = ['T3S', 'TrjSR', 'TrajGAT', 'TrajCL', 'NEUTRAJ', 'MLP1', 'MLP2'] # TMN is not applicable
    for dbset_size in dbset_sizes:
        for method_name in methods:
            knnindex = KnnQuery(method_name)
            
            try:
                build_metrics = knnindex.build_index(dbset_size, dbset_trajlen_min, dbset_trajlen_max)
            except Exception as excpt:
                del knnindex
                torch.cuda.empty_cache()
                gc.collect()
                logging.error("[ERROR].index construction error. dataset={},"
                                "dbset_size={},qset_size={},min_traj_len={},max_traj_len={},error={}--{}".format( \
                                Config.dataset, dbset_size, qset_size, qset_trajlen_min, qset_trajlen_max, type(excpt), excpt))
                continue
            
            for _isgpu in [True, False]:
                Config.gpu = _isgpu
                query_metrics = knnindex.query(Config.knn_topk, qset_size, qset_trajlen_min, qset_trajlen_max)
                
                logging.info('[EXPFlag]exp=knn_dbset_sizes,fn={},dataset={},gpu={},'
                            'dbset_size={},qset_size={},min_traj_len={},max_traj_len={},'
                            'buildtime={:.4f},buildmem={},'
                            'querytime={:.4f},querymem={},querytimeemb={:.4f}'.format( \
                            method_name, Config.dataset, Config.gpu,
                            dbset_size, qset_size, qset_trajlen_min, qset_trajlen_max,
                            build_metrics['time'], build_metrics['mem'], 
                            query_metrics['time'], query_metrics['mem'], query_metrics['time_emb'] ))

            del knnindex
            torch.cuda.empty_cache()
            gc.collect()
    
    return


def test_varying_qset_numpoints():
    if Config.gpu:
        faiss.omp_set_num_threads(Config.knn_query_threads_gpu)
    else:  
        faiss.omp_set_num_threads(Config.knn_query_threads_cpu)

    dbset_size = 10**5
    dbset_trajlen_min = 20
    dbset_trajlen_max = 800
    qset_size = 1000
    qset_trajlens = [(20, 200), (201,400), (401,800), (801,1600)]

    methods = ['T3S', 'TrjSR', 'TrajGAT', 'TrajCL', 'NEUTRAJ', 'MLP1', 'MLP2'] # TMN is not applicable
    for method_name in methods:
        knnindex = KnnQuery(method_name)
        
        try:
            build_metrics = knnindex.build_index(dbset_size, dbset_trajlen_min, dbset_trajlen_max)
        except Exception as excpt:
            del knnindex
            torch.cuda.empty_cache()
            gc.collect()
            logging.error("[ERROR].index construction error. dataset={},"
                            "dbset_size={},qset_size={},min_traj_len={},max_traj_len={},error={}--{}".format( \
                            Config.dataset, dbset_size, qset_size, qset_trajlen_min, qset_trajlen_max, type(excpt), excpt))
            continue
                    
        for (qset_trajlen_min, qset_trajlen_max) in qset_trajlens:
            for _isgpu in [True, False]:
                Config.gpu = _isgpu
                query_metrics = knnindex.query(Config.knn_topk, qset_size, qset_trajlen_min, qset_trajlen_max)
                
                logging.info('[EXPFlag]exp=knn_qset_numpoints,fn={},dataset={},gpu={},'
                            'dbset_size={},qset_size={},min_traj_len={},max_traj_len={},'
                            'buildtime={:.4f},buildmem={},'
                            'querytime={:.4f},querymem={},querytimeemb={:.4f}'.format( \
                            method_name, Config.dataset, Config.gpu,
                            dbset_size, qset_size, qset_trajlen_min, qset_trajlen_max,
                            build_metrics['time'], build_metrics['mem'], 
                            query_metrics['time'], query_metrics['mem'], query_metrics['time_emb'] ))

        del knnindex
        torch.cuda.empty_cache()
        gc.collect()
    return


class KnnQuery:
    def __init__(self, method_name):
        super(KnnQuery, self).__init__()
        
        self.method_name = method_name
        self.model = None
        self.dataloader_collate_fn = None
        
        self.index = None
    
    def __del__(self):
        if self.index:
            del self.index
        gc.collect()

    def build_index(self, dbset_size, trajlen_min, trajlen_max):
        dataset = DatasetST_h5(Config.dataset_file+'.h5', False, dbset_size, trajlen_min, trajlen_max, 2000)
        
        _time = time.time()
        _mem = tool_funcs.RAMInfo.mem()
        device = torch.device('cuda:0')
        batch_size = Config.knn_emb_batch_size_gpu 
        db_traj_embs = self.read_traj_file_and_emb_generator(dataset, batch_size, device)

        faiss.omp_set_num_threads(1)
        indextype = Config.knn_faiss_index
        d = Config.traj_embedding_dim
        
        if indextype == 'HNSW':
            M = 32
            self.index = faiss.IndexHNSWFlat(d, M)
        elif indextype == 'LSH':
            nbits = 4
            self.index = faiss.IndexLSH(d, nbits)
        elif indextype == 'FLATL1':
            self.index = faiss.IndexFlat(d, faiss.METRIC_L1)
        elif indextype == 'IVF' or True:
            nlist = 100
            quantizer = faiss.IndexFlat(d, faiss.METRIC_L1)   # build the index
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist)
            self.index.nprobe = 10
            self.index.train(db_traj_embs)
            
        self.index.add(db_traj_embs)
        _endtime = time.time()
        _endmem = tool_funcs.RAMInfo.mem()
        logging.debug("[build_index] done. @={}, mem={}/{}".format(_endtime-_time, _endmem, _endmem-_mem))

        del dataset
        del db_traj_embs
        torch.cuda.empty_cache()
        gc.collect()

        return {'time': _endtime-_time, 'mem': _endmem-_mem}


    def query(self, k, qset_size, trajlen_min, trajlen_max):
        datafile = Config.dataset_file+'.h5'
        dataset = DatasetST_h5(Config.dataset_file+'.h5', False, qset_size, trajlen_min, trajlen_max, 2001) # use diff seed

        _time = time.time()
        _mem = tool_funcs.RAMInfo.mem()
        device = torch.device('cuda:0') if Config.gpu else torch.device('cpu')
        batch_size = Config.knn_emb_batch_size_gpu if Config.gpu else Config.knn_emb_batch_size_cpu
        q_traj_embs = self.read_traj_file_and_emb_generator(dataset, batch_size, device)
        _time_emb = time.time()
        
        D, I = self.index.search(q_traj_embs, k)
       
        _endtime = time.time()
        _endmem = tool_funcs.RAMInfo.mem()
        logging.debug("[query] done. @={}, mem={}/{}".format(_endtime-_time, _endmem, _endmem-_mem))
        logging.debug(I[:5,:10])
        
        del dataset
        del q_traj_embs
        torch.cuda.empty_cache()
        gc.collect()

        return {'time': _endtime-_time, 'time_emb': _time_emb-_time, 'mem': _endmem-_mem}
    
    
    # return trajemb on cpu as numpy array
    @torch.no_grad()
    def read_traj_file_and_emb_generator(self, dataset, batch_size, device):
        _time = time.time()
        
        merc_range = dataset.merc_range
        if self.model == None:
            self.model = learned_class_wrapper(self.method_name, merc_range)
        self.model.eval()
        self.model.to(device)
        interpret_fn = self.model.interpret
        if self.dataloader_collate_fn == None:
            self.dataloader_collate_fn = learned_collate_fn_wrapper(self.method_name, self.model, False)
        
        dataloader = DataLoader(dataset, batch_size = batch_size, 
                                shuffle=False, drop_last = False, 
                                collate_fn = lambda x: self.dataloader_collate_fn(list(zip(*x))[0]), 
                                num_workers = Config.knn_dataloader_num_workers)
        n = len(dataset)
        traj_embs = np.empty([n, Config.traj_embedding_dim], dtype = np.float32)

        for i, batch_after_collate in enumerate(dataloader):
            embs = interpret_fn(batch_after_collate).cpu().detach().numpy()
            traj_embs[i*batch_size: i*batch_size+len(embs), :] = embs
        
        logging.debug("Raw traj -> embs. done. #={}, @={:.3f}".format(n, time.time()-_time))
        return traj_embs 
    
    
def parse_args():
    parser = argparse.ArgumentParser(description = "...")
    # Font give default values here, because they will be faultly 
    # overwriten by the values in config.py.
    # config.py is the correct place for default values
    
    parser.add_argument('--dumpfile_uniqueid', type = str, help = '') # see config.py
    parser.add_argument('--seed', type = int, help = '')
    parser.add_argument('--gpu', dest = 'gpu', action='store_true')
    parser.add_argument('--dataset', type = str, help = '')
    parser.add_argument('--knn_exp', type = str, help = 'dbset_sizes|qset_numpoints', required=True)
    parser.add_argument('--knn_faiss_index', type = str, help = '')
    
    parser.add_argument('--knn_emb_batch_size_gpu', type = int, help = '')
    
    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))



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
    
    starting_time = time.time()
    
    if Config.knn_exp == 'dbset_sizes':
        test_varying_dbset_sizes()
    elif Config.knn_exp == 'qset_numpoints':
        test_varying_qset_numpoints()
    
    logging.info('all finished. @={:.1f}'.format(time.time() - starting_time))
    