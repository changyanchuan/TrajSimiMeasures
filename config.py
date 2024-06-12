import os


class Config:
    
    debug = False
    dumpfile_uniqueid = ''
    seed = 2000
    gpu = False
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    root_dir = os.path.dirname(os.path.abspath(__file__)) # ./
    data_dir = os.path.join(root_dir, 'data') # ./data
    snapshot_dir = os.path.join(root_dir, 'exp', 'snapshot')
    
    dataset = ''
    dataset_prefix = ''
    dataset_file = ''
    dataset_trajsimi_dict = ''
    
    min_lon = 0.0
    min_lat = 0.0
    max_lon = 0.0
    max_lat = 0.0
    cell_size = 100
    
    # dataset preprocessing
    min_traj_len = 20
    max_traj_len = 2147483647
    
    # trajsimi_efficiency
    effi_exp = 'numtrajs' # specific exp
    effi_method = ''
    effi_auxilary_processor = True
    effi_num_trajs = 100000
    effi_min_traj_len = 20
    effi_max_traj_len = 200
    effi_dataloader_num_workers = 0
    effi_cpu_method_num_cores = 8
    effi_batch_size_cpu_heuristic = 512
    effi_batch_size_cpu_learned = 512
    effi_batch_size_gpu = 512
    effi_timeout = 7200
    effi_gpu_threads_per_traj = 64
    
    
    # trajsimi effectivness experiments
    trajsimi_measure = ''
    trajsimi_min_traj_len = 20
    trajsimi_max_traj_len = 200
    trajsimi_edr_lcss_eps = 200
    trajsimi_edr_lcss_delta = 60
    trajsimi_timereport_exp = False
    trajsimi_sar_distance_eps = 200 # same to trajsimi_edr_lcss_eps 
    trajsimi_sar_time_eps = 10 
    trajsimi_sar_target_length = 10
    
    
    # knn query exp
    knn_exp = ''
    knn_topk = 50
    knn_dataloader_num_workers = 0
    knn_query_threads_cpu = 1
    knn_query_threads_gpu = 8
    knn_emb_batch_size_cpu = 1 # for learned methods
    knn_emb_batch_size_gpu = 512
    knn_faiss_index = 'IVF' # [IVF, HNSW, LSH, FLATL1]
    
    # clustering exp
    clustering_exp = 'numtrajs'
    clustering_num_centroids = 10
    clustering_method = ''
    clustering_truth_heur_method = 'dtw'
    clustering_num_trajs = 1000
    clustering_min_traj_len = 20
    clustering_max_traj_len = 200
    clustering_auxilary_processor = True

    
    # learned methods
    cell_embedding_dim = 128
    traj_embedding_dim = 128
    
    # TrjSR
    trjsr_imgsize_x_lr = 162
    trjsr_imgsize_y_lr = 128
    trjsr_pixelrange_lr = 2
    
    # TMN
    tmn_pooling_size = 10
    tmn_sampling_num = 20
    
    # TrajGAT
    trajgat_num_head = 8
    trajgat_num_encoder_layers = 3
    trajgat_d_lap_pos = 8
    trajgat_encoder_dropout = 0.01
    trajgat_dataloader_num_workers = 8
    trajgat_qtree_node_capacity = 50
    
    # RSTS
    rsts_num_layers = 3
    rsts_bidirectional = True
    rsts_dropout = 0.2
    
    @classmethod
    def post_value_updates(cls):

        if 'xian' == cls.dataset:
            cls.dataset_prefix = 'xian_7_20inf'
            cls.min_lon = 108.92185
            cls.min_lat = 34.20494
            cls.max_lon = 109.00884
            cls.max_lat = 34.2787
        
        elif 'porto' == cls.dataset:
            cls.dataset_prefix = 'porto_20inf'
            cls.min_lon = -8.7005
            cls.min_lat = 41.1001
            cls.max_lon = -8.5192
            cls.max_lat = 41.2086
            # -15.630759, -3.930948, 36.886104, 45.657225 
            # all raw trajectories in porto 
            
        elif 'geolife' == cls.dataset: # geolife_all
            cls.dataset_prefix = 'geolife_all_20inf'
            cls.min_lon = 115.4187
            cls.min_lat = 39.4416
            cls.max_lon = 117.5081
            cls.max_lat = 41.0598
            
        elif 'chengdu' == cls.dataset:
            cls.dataset_prefix = 'chengdu_200k_20inf'
            cls.min_lon = 104.04214
            cls.min_lat = 30.65294
            cls.max_lon = 104.12958
            cls.max_lat = 30.72775

        elif 'germany' == cls.dataset:
            cls.dataset_prefix = 'germany_20200'
            cls.min_lon = 5.85
            cls.min_lat = 47.25
            cls.max_lon = 15.05
            cls.max_lat = 54.00
            cls.cell_size = 1000
            
        else:
            pass
        
        cls.dataset_file = os.path.join(cls.data_dir, cls.dataset_prefix)
        
        cls.dataset_trajsimi_traj = '{}_trajsimi_{}{}_dict_traj'.format( \
                                    cls.dataset_file, cls.trajsimi_min_traj_len, 
                                    cls.trajsimi_max_traj_len)
        
        cls.dataset_trajsimi_dict = '{}_trajsimi_{}{}_dict_{}'.format( \
                                    cls.dataset_file, cls.trajsimi_min_traj_len, 
                                    cls.trajsimi_max_traj_len, 
                                    cls.trajsimi_measure)

    @classmethod
    def update(cls, dic: dict):
        for k, v in dic.items():
            if k in cls.__dict__:
                assert type(getattr(Config, k)) == type(v)
            setattr(Config, k, v)
        cls.post_value_updates()


    @classmethod
    def to_str(cls): # __str__, self
        dic = cls.__dict__.copy()
        lst = list(filter( \
                        lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                        dic.items() \
                        ))
        return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])

