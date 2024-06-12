import sys
sys.path.append('..')
sys.path.append('../..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
import pickle
import argparse

from config import Config as Config
from utilities import tool_funcs
from utils.cellspace import CellSpace
from TMN import TMN
from NEUTRAJ_utils import distance_sampling, negative_distance_sampling, pad_sequence

tmn_grid_delta = 100
tmn_epochs = 20
tmn_batch_size = 10
tmn_learning_rate = 0.001
tmn_training_bad_patience = 5
tmn_batch_size_testing = 100



class TMNTrainer:
    def __init__(self):
        super(TMNTrainer, self).__init__()
        self.device = torch.device('cuda:0')
        x_min, y_min = tool_funcs.lonlat2meters(Config.min_lon, Config.min_lat)
        x_max, y_max = tool_funcs.lonlat2meters(Config.max_lon, Config.max_lat)
        self.cellspace = CellSpace(Config.cell_size, Config.cell_size, 
                                    x_min, y_min, x_max, y_max)

        self.model = TMN(4, Config.traj_embedding_dim, (self.cellspace.x_size, self.cellspace.y_size), 
                    Config.tmn_sampling_num, (x_min, x_max), (y_min, y_max), Config.cell_size)
        self.model.to(self.device)
        
        self.dic_datasets = self.load_trajsimi_dataset()
        
        self.checkpoint_path = '{}/{}_trajsimi_TMN_{}_best{}.pt'.format(Config.snapshot_dir, \
                                    Config.dataset_prefix, Config.trajsimi_measure, Config.dumpfile_uniqueid)


    def train(self):
        logging.info("training. START! @={:.3f}".format(time.time()))
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        torch.autograd.set_detect_anomaly(True)
        
        self.criterion = WeightedRankingLoss(batch_size = tmn_batch_size, sampling_num = Config.tmn_sampling_num)
        self.criterion.to(self.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), \
                                            lr = tmn_learning_rate)

        best_hr_eval = 0.0
        best_loss_train = 10000000.0
        best_epoch = 0
        bad_counter = 0
        bad_patience = tmn_training_bad_patience
        timetoreport = [1200, 2400, 3600] # len may change later

        for i_ep in range(tmn_epochs):
            _time_ep = time.time()
            train_losses = []
            train_gpu = []
            train_ram = []

            self.model.train()

            for i_batch, batch in enumerate( self.trajsimi_dataset_generator_pairs_batchi( \
                                                            self.dic_datasets['trains_trajcoorgrid'], \
                                                            self.dic_datasets['trains_simi'], \
                                                            self.dic_datasets['max_distance'])):
                _time_batch = time.time()
                optimizer.zero_grad()
                
                inputs_arrays, inputs_len_arrays, distance_arrays = batch[0], batch[1], batch[2]

                trajs_loss, negative_loss = self.model(inputs_arrays, inputs_len_arrays)

                positive_distance_target = torch.tensor(distance_arrays[0], device = self.device).view((-1, 1))
                negative_distance_target = torch.tensor(distance_arrays[1], device = self.device).view((-1, 1))

                train_loss = self.criterion(trajs_loss, positive_distance_target,
                                    negative_loss, negative_distance_target)

                train_loss.backward()
                optimizer.step()

                train_losses.append(train_loss.item())
                train_gpu.append(tool_funcs.GPUInfo.mem()[0])
                train_ram.append(tool_funcs.RAMInfo.mem())

                # debug output
                if i_batch % 100 == 0 and Config.debug:
                    logging.debug("training. ep-batch={}-{}, train_loss={:.4f}, @={:.3f}, gpu={}, ram={}" \
                                .format(i_ep, i_batch, train_loss.item(), 
                                        time.time()-_time_batch, tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))

                # exp of training time vs. effectiveness
                if Config.trajsimi_timereport_exp and len(timetoreport) \
                        and time.time() - training_starttime >= timetoreport[0]:
                    test_metrics = self.__test(self.dic_datasets['tests_trajcoorgrid'], \
                                                self.dic_datasets['tests_simi'], \
                                                self.dic_datasets['max_distance'])
                    logging.info("test.  ts={}, hr={:.3f},{:.3f},{:.3f}".format(timetoreport[0], *test_metrics))
                    timetoreport.pop(0)
                    self.model.train()

            # ep debug output
            logging.info("training. i_ep={}, loss={:.4f}, @={:.3f}" \
                        .format(i_ep, tool_funcs.mean(train_losses), time.time()-_time_ep))
            
            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)

            # eval
            eval_metrics = self.__test(self.dic_datasets['evals_trajcoorgrid'], \
                                        self.dic_datasets['evals_simi'], \
                                        self.dic_datasets['max_distance'])
            logging.info("eval.     i_ep={}, hr={:.3f},{:.3f},{:.3f}".format(i_ep, *eval_metrics))
            eval_hr_ep = eval_metrics[0]

            # early stopping
            if eval_hr_ep > best_hr_eval:
                best_epoch = i_ep
                best_hr_eval = eval_hr_ep
                best_loss_train = tool_funcs.mean(train_losses)
                bad_counter = 0
                torch.save({'model': self.model.state_dict()}, self.checkpoint_path)
            else:
                bad_counter += 1

            if bad_counter == bad_patience or i_ep + 1 == tmn_epochs:
                training_endtime = time.time()
                logging.info("training end. @={:.0f}, best_epoch={}, best_loss_train={:.4f}, best_hr_eval={:.4f}, #param={}" \
                            .format(training_endtime - training_starttime, \
                                    best_epoch, best_loss_train, best_hr_eval, \
                                    tool_funcs.num_of_model_params(self.model) ))
                break
            
        # test
        test_starttime = time.time()
        test_metrics = self.__test(self.dic_datasets['tests_trajcoorgrid'], \
                                    self.dic_datasets['tests_simi'], \
                                    self.dic_datasets['max_distance'])
        test_endtime = time.time()
        logging.info("test.     hr={:.3f},{:.3f},{:.3f}".format(*test_metrics))
        return {'task_train_time': training_endtime - training_starttime, \
                'task_train_gpu': training_gpu_usage, \
                'task_train_ram': training_ram_usage, \
                'task_test_time': test_endtime - test_starttime, \
                'task_test_gpu': 0, \
                'task_test_ram': 0, \
                'hr10': test_metrics[0], 'hr50': test_metrics[1], 'hr50in10': test_metrics[2]}
    

    # inner calling only
    @torch.no_grad()
    def __test(self, trajs_coorgrid, datasets_simi, max_distance):
        self.model.eval()
        
        dists = []
        datasets_simi = torch.tensor(datasets_simi, device = self.device, dtype = torch.float) / max_distance

        for _, batch in enumerate(self.trajsimi_dataset_generator_batchi(trajs_coorgrid)):
            inputs_arrays, inputs_len_arrays = batch
            inputs0 = torch.Tensor(inputs_arrays[0]).to(self.device)
            inputs1 = torch.Tensor(inputs_arrays[1]).to(self.device)
            embs0, embs1 = self.model.smn.f(inputs0, inputs_len_arrays[0], inputs1, inputs_len_arrays[1])
            dist = torch.exp(F.pairwise_distance(embs0, embs1, p = 2))
            dists.extend(dist)
        
        dists = torch.tensor(dists, dtype = torch.float)
        pred_l1_simi = torch.zeros(datasets_simi.shape)
        triu_idx = torch.triu_indices(datasets_simi.shape[0], datasets_simi.shape[1], 1)
        pred_l1_simi[triu_idx[0,:], triu_idx[1,:]] = dists
        pred_l1_simi = pred_l1_simi + pred_l1_simi.T
        truth_l1_simi = datasets_simi

        hrA = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 10, 10)
        hrB = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 50, 50)
        hrBinA = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 50, 10)

        return hrA, hrB, hrBinA
        

    def trajsimi_dataset_generator_batchi(self, trajs_coorgrid):
        i = 0 
        j = 1
        n = len(trajs_coorgrid)
        trajs_len = [len(traj) for traj in trajs_coorgrid]
        
        while True:
            if i >= n - 1:
                break
            anchor_input, trajs_input = [], []
            anchor_input_len, trajs_input_len = [], []
            while True:
                anchor_input.append(trajs_coorgrid[i])
                trajs_input.append(trajs_coorgrid[j])
                anchor_input_len.append(trajs_len[i])
                trajs_input_len.append(trajs_len[j])
                
                j += 1
                if j == n:
                    i += 1
                    j = i + 1
                if i == n - 1 or len(anchor_input) == tmn_batch_size:
                    break
                
            max_anchor_length = max(anchor_input_len)
            max_sample_lenght = max(trajs_input_len)
            anchor_input = pad_sequence(anchor_input, maxlen=max_anchor_length)
            trajs_input = pad_sequence(trajs_input, maxlen=max_sample_lenght)

            yield ([np.array(anchor_input),np.array(trajs_input)],
                   [anchor_input_len, trajs_input_len])

    def trajsimi_dataset_generator_pairs_batchi(self, trajs_coorgrid, datasets_simi, max_distance):
        len_datasets = len(trajs_coorgrid)
        # datasets_simi = torch.tensor(datasets_simi, device = self.device, dtype = torch.float) / max_distance
        datasets_simi = np.asarray(datasets_simi, dtype = np.float32) / max_distance
        trajs_len = [len(traj) for traj in trajs_coorgrid]
        
        cur_index = 0
        while cur_index < len_datasets:
            end_index = cur_index + tmn_batch_size if cur_index + tmn_batch_size < len_datasets else len_datasets
            
            anchor_input, trajs_input, negative_input, distance, negative_distance = [], [], [], [], []
            anchor_input_len, trajs_input_len, negative_input_len = [], [], []
            batch_trajs_keys = {}
            batch_trajs_input, batch_trajs_len = [], []

            for _i in range(cur_index, end_index):

                sampling_index_list = distance_sampling(datasets_simi, len_datasets, _i, Config.tmn_sampling_num)
                negative_sampling_index_list = negative_distance_sampling(datasets_simi, len_datasets, _i, Config.tmn_sampling_num)

                trajs_input.append(trajs_coorgrid[_i]) #
                anchor_input.append(trajs_coorgrid[_i])
                negative_input.append(trajs_coorgrid[_i])
                if _i not in batch_trajs_keys:
                    batch_trajs_keys[_i] = 0
                    batch_trajs_input.append(trajs_coorgrid[_i])
                    batch_trajs_len.append(trajs_len[_i])

                anchor_input_len.append(trajs_len[_i])
                trajs_input_len.append(trajs_len[_i])
                negative_input_len.append(trajs_len[_i])

                distance.append(1)
                negative_distance.append(1)

                for traj_index in sampling_index_list:

                    anchor_input.append(trajs_coorgrid[_i])
                    trajs_input.append(trajs_coorgrid[traj_index])

                    anchor_input_len.append(trajs_len[_i])
                    trajs_input_len.append(trajs_len[traj_index])

                    if traj_index not in batch_trajs_keys:
                        batch_trajs_keys[_i] = 0
                        batch_trajs_input.append(trajs_coorgrid[traj_index])
                        batch_trajs_len.append(trajs_len[traj_index])

                    # distance.append(np.exp(-float(datasets_simi[_i][traj_index])*Config.tmn_sampling_num))
                    distance.append(float(datasets_simi[_i][traj_index]))

                for traj_index in negative_sampling_index_list:

                    negative_input.append(trajs_coorgrid[traj_index])
                    negative_input_len.append(trajs_len[traj_index])
                    # negative_distance.append(np.exp(-float(datasets_simi[_i][traj_index])*Config.tmn_sampling_num))
                    negative_distance.append(float(datasets_simi[_i][traj_index]))

                    if traj_index not in batch_trajs_keys:
                        batch_trajs_keys[_i] = 0
                        batch_trajs_input.append(trajs_coorgrid[traj_index])
                        batch_trajs_len.append(trajs_len[traj_index])
            
            max_anchor_length = max(anchor_input_len)
            max_sample_lenght = max(trajs_input_len)
            max_neg_lenght = max(negative_input_len)
            anchor_input = pad_sequence(anchor_input, maxlen=max_anchor_length)
            trajs_input = pad_sequence(trajs_input, maxlen=max_sample_lenght)
            negative_input = pad_sequence(negative_input, maxlen=max_neg_lenght)
            batch_trajs_input = pad_sequence(batch_trajs_input, maxlen=max(max_anchor_length, max_sample_lenght,
                                                                           max_neg_lenght))

            yield ([np.array(anchor_input),np.array(trajs_input),np.array(negative_input)],
                   [anchor_input_len, trajs_input_len, negative_input_len],
                   [np.array(distance),np.array(negative_distance)])
            cur_index = end_index




    def load_trajsimi_dataset(self):
        # 1. read TrajSimi dataset
        # 2. convert merc_seq to cell_id_seqs
        
        with open(Config.dataset_trajsimi_traj, 'rb') as fh:
            dic_dataset = pickle.load(fh)
            trajs_merc = dic_dataset['trajs_merc']
            
            
        with open(Config.dataset_trajsimi_dict, 'rb') as fh:
            dic_dataset = pickle.load(fh)
            train_simis = dic_dataset['train_simis']
            eval_simis = dic_dataset['eval_simis']
            test_simis = dic_dataset['test_simis']
            max_distance = dic_dataset['max_distance']
            
            trains_trajcoorgrid = self.__trajcoor_to_trajcoor_and_trajgrid(trajs_merc[:7000])
            evals_trajcoorgrid = self.__trajcoor_to_trajcoor_and_trajgrid(trajs_merc[7000:8000])
            tests_trajcoorgrid = self.__trajcoor_to_trajcoor_and_trajgrid(trajs_merc[8000:])

        logging.info("trajsimi dataset sizes. (trains/evals/tests={}/{}/{})" \
                    .format(len(trains_trajcoorgrid), len(evals_trajcoorgrid), len(tests_trajcoorgrid)))

        return {'trains_trajcoorgrid': trains_trajcoorgrid, 'evals_trajcoorgrid': evals_trajcoorgrid, \
                'tests_trajcoorgrid': tests_trajcoorgrid, \
                'trains_simi': train_simis, 'evals_simi': eval_simis, 'tests_simi': test_simis, \
                'max_distance': max_distance}


    def __trajcoor_to_trajcoor_and_trajgrid(self, lst_trajs: list):
        # lst_trajs = [ [[lon, lat_in_merc_space] ,[] ], ..., [], ..., ] 

        logging.info("trajcoor_to_trajgrid starts. #={}".format(len(lst_trajs)))
        _time = time.time()

        lst_trajs_nodes_gridid = []
        for traj in lst_trajs:
            traj_nodes_gridid = []
            for xy in traj:
                i_x = int( (xy[0] - self.cellspace.x_min) / self.cellspace.x_unit )
                i_y = int( (xy[1] - self.cellspace.y_min) / self.cellspace.y_unit )
                i_x = i_x - 1 if i_x == self.cellspace.x_size else i_x
                i_y = i_y - 1 if i_y == self.cellspace.y_size else i_y
                traj_nodes_gridid.append( (i_x, i_y) )
            lst_trajs_nodes_gridid.append( traj_nodes_gridid )       

        # local normlization
        xs, ys = zip(*[[p[0], p[1]] for traj in lst_trajs for p in traj])
        meanx, meany, stdx, stdy = np.mean(xs), np.mean(ys), np.std(xs), np.std(ys)
        lst_trajs_nodes_lonlat = [[[(p[0] - meanx) / stdx, (p[1] - meany) / stdy] for p in traj] for traj in lst_trajs]

        lst_trajs_node_lonlat_gridid = []
        for i in range(len(lst_trajs_nodes_lonlat)):
            traj = []
            for coor, grid in zip(lst_trajs_nodes_lonlat[i], lst_trajs_nodes_gridid[i]):
                traj.append([coor[0], coor[1], grid[0], grid[1]])
            lst_trajs_node_lonlat_gridid.append(traj)  
              

        logging.info("trajcoor_to_trajgrid ends. @={:.3f}, #={}" \
                    .format(time.time() - _time, len(lst_trajs_node_lonlat_gridid)))
        # lst_trajs_nodes_lonlat = [ [[lon, lat_normalized] ,[] ], ..., [], ..., ] 
        # lst_trajs_nodes_gridid = [ [gridid, gridid, ... ], ..., [], ..., ] 
        # return lst_trajs_nodes_lonlat, lst_trajs_nodes_gridid
        return lst_trajs_node_lonlat_gridid



class WeightedRankingLoss(nn.Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightedRankingLoss, self).__init__()
        self.positive_loss = WeightMSELoss(batch_size, sampling_num)
        self.negative_loss = WeightMSELoss(batch_size, sampling_num)

    def forward(self, p_input, p_target, n_input, n_target):
        # trajs_mse_loss = self.positive_loss(p_input, autograd.Variable(p_target).to(Config.device), False)
        trajs_mse_loss = self.positive_loss(p_input, p_target, False)

        # negative_mse_loss = self.negative_loss(n_input, autograd.Variable(n_target).to(Config.device), True)
        negative_mse_loss = self.negative_loss(n_input, n_target, True)

        self.trajs_mse_loss = trajs_mse_loss
        self.negative_mse_loss = negative_mse_loss
        loss = sum([trajs_mse_loss,negative_mse_loss])
        return loss


class WeightMSELoss(nn.Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightMSELoss, self).__init__()
        # self.weight = []
        # for i in range(batch_size):
        #     self.weight.append(0.)
        #     for traj_index in range(sampling_num):
        #         self.weight.append(np.array([sampling_num - traj_index]))

        # self.weight = np.array(self.weight)
        # sum = np.sum(self.weight)
        # self.weight = self.weight / sum
        # self.weight = self.weight.astype(np.float32)
        # self.weight = Parameter(torch.Tensor(self.weight).to(Config.device), requires_grad = False)
        weight_lst = [0] + list(range(sampling_num, 0, -1))
        self.register_buffer('weight', torch.tensor(weight_lst, dtype = torch.float, requires_grad = False))

        # self.batch_size = batch_size
        self.sampling_num = sampling_num

    def forward(self, input, target, isReLU = False):
        # div = target - input.view(-1,1)
        # if isReLU:
        #     div = F.relu(div.view(-1,1))
        # square = torch.mul(div.view(-1,1), div.view(-1,1))
        # weight_square = torch.mul(square.view(-1,1), self.weight) # self.weight.view(-1,1))
        div = target.squeeze(-1) - input
        if isReLU:
            div = F.relu(div)
        square = torch.mul(div, div)
        weight = self.weight.repeat(square.shape[0] // self.weight.shape[0])
        weight_square = torch.mul(square, weight / weight.sum())
        loss = torch.sum(weight_square)
        return loss


def parse_args():
    parser = argparse.ArgumentParser(description = "...")
    # dont give default value here! Otherwise, it will faultly overwrite the value in config.py.
    # config.py is the correct place to provide default values
    parser.add_argument('--debug', dest = 'debug', action='store_true')
    parser.add_argument('--dumpfile_uniqueid', type = str, help = '') # see config.py
    parser.add_argument('--seed', type = int, help = '')
    parser.add_argument('--dataset', type = str, help = '')
    parser.add_argument('--trajsimi_measure', type = str, help = '')
    
    parser.add_argument('--cell_embedding_dim', type = int, help = '')
    parser.add_argument('--traj_embedding_dim', type = int, help = '')
    parser.add_argument('--trajsimi_min_traj_len', type = int, help = '')
    parser.add_argument('--trajsimi_max_traj_len', type = int, help = '')
    
    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


# nohup python TMN_trajsimi_train.py --dataset xian --trajsimi_measure dtw --seed 2000 --debug &> ../result &
if __name__ == '__main__':
    
    Config.update(parse_args())
    logging.basicConfig(level = logging.DEBUG,
            format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
            handlers = [logging.FileHandler(Config.root_dir+'/exp/log/'+tool_funcs.log_file_name(), mode = 'w'), 
                        logging.StreamHandler()]
            )

    logging.info('python ' + ' '.join(sys.argv))
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    tool_funcs.set_seed(Config.seed)

    tmn = TMNTrainer()
    tmn.train()

