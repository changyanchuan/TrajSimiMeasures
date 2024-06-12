import sys
sys.path.append('..')
sys.path.append('../..')

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import time
import logging
import math
import random
import pickle
import argparse

from config import Config as Config
from utilities import tool_funcs
from MLP import MLP


mlp_epochs = 20
mlp_batch_size = 256
mlp_learning_rate = 0.0002
mlp_training_bad_patience = 5


class MLPTrainer:
    def __init__(self):
        super(MLPTrainer, self).__init__()
        self.device = torch.device('cuda:0')

        self.dic_datasets = self.load_trajsimi_dataset()
        
        self.checkpoint_path = '{}/{}_trajsimi_MLP_{}_best{}.pt'.format(Config.snapshot_dir, \
                                    Config.dataset_prefix, Config.trajsimi_measure, Config.dumpfile_uniqueid)


    def train(self):
        logging.info("training. START! @={:.3f}".format(time.time()))
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        torch.autograd.set_detect_anomaly(True)
        
        
        self.model = MLP(2, Config.traj_embedding_dim)
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.criterion.to(self.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), \
                                            lr = mlp_learning_rate)

        best_hr_eval = 0.0
        best_loss_train = 10000000.0
        best_epoch = 0
        bad_counter = 0
        bad_patience = mlp_training_bad_patience

        for i_ep in range(mlp_epochs):
            _time_ep = time.time()
            train_losses = []
            train_gpu = []
            train_ram = []

            self.model.train()

            for i_batch, batch in enumerate( self.trajsimi_dataset_generator_pairs_batchi( \
                                                            self.dic_datasets['trains_trajcoor'], \
                                                            self.dic_datasets['trains_simi'], \
                                                            self.dic_datasets['max_distance'])):
                _time_batch = time.time()
                optimizer.zero_grad()

                sub_trajs_coor, sub_trajs_len, sub_simi = batch
                _max_sub_trajs_len = max(sub_trajs_len) 
                src_padding_mask = torch.arange(_max_sub_trajs_len, device = self.device)[None, :] >= sub_trajs_len[:, None]

                outs = self.model(sub_trajs_coor, src_padding_mask, sub_trajs_len)
                pred_l1_simi = torch.cdist(outs, outs, 1)
                pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
                truth_l1_simi = sub_simi[torch.triu(torch.ones(sub_simi.shape), diagonal = 1) == 1]
                train_loss = self.criterion(pred_l1_simi, truth_l1_simi)
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

            # ep debug output
            logging.info("training. i_ep={}, loss={:.4f}, @={:.3f}" \
                        .format(i_ep, tool_funcs.mean(train_losses), time.time()-_time_ep))
            
            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)

            # eval
            eval_metrics = self.__test(self.dic_datasets['evals_trajcoor'], \
                                        self.dic_datasets['evals_simi'], \
                                        self.dic_datasets['max_distance'])
            logging.info("eval.     i_ep={}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f}".format(i_ep, *eval_metrics))
            eval_hr_ep = eval_metrics[1]

            # early stopping
            if eval_hr_ep > best_hr_eval:
                best_epoch = i_ep
                best_hr_eval = eval_hr_ep
                best_loss_train = tool_funcs.mean(train_losses)
                bad_counter = 0
                torch.save({'model': self.model.state_dict()}, self.checkpoint_path)
            else:
                bad_counter += 1

            if bad_counter == bad_patience or i_ep + 1 == mlp_epochs:
                training_endtime = time.time()
                logging.info("training end. @={:.0f}, best_epoch={}, best_loss_train={:.4f}, best_hr_eval={:.4f}, #param={}" \
                            .format(training_endtime - training_starttime, \
                                    best_epoch, best_loss_train, best_hr_eval, \
                                    tool_funcs.num_of_model_params(self.model) ))
                break
            
        # test
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        test_starttime = time.time()
        test_metrics = self.__test(self.dic_datasets['tests_trajcoor'], \
                                    self.dic_datasets['tests_simi'], \
                                    self.dic_datasets['max_distance'])
        test_endtime = time.time()
        logging.info("test.     loss= {:.4f}, hr={:.3f},{:.3f},{:.3f}".format(*test_metrics))
        return {'task_train_time': training_endtime - training_starttime, \
                'task_train_gpu': training_gpu_usage, \
                'task_train_ram': training_ram_usage, \
                'task_test_time': test_endtime - test_starttime, \
                'task_test_gpu': 0, \
                'task_test_ram': 0, \
                'hr10': test_metrics[1], 'hr50': test_metrics[2], 'hr50in10': test_metrics[3]}
    

    # inner calling only
    @torch.no_grad()
    def __test(self, trajs_coor, datasets_simi, max_distance):
        self.model.eval()
        
        traj_embs = []
        datasets_simi = torch.tensor(datasets_simi, device = self.device, dtype = torch.float)
        datasets_simi = (datasets_simi + datasets_simi.T) / max_distance

        for i_batch, batch in enumerate(self.trajsimi_dataset_generator_batchi(trajs_coor)):
            sub_trajs_coor, sub_trajs_len = batch
            _max_sub_trajs_len = max(sub_trajs_len) 
            src_padding_mask = torch.arange(_max_sub_trajs_len, device = self.device)[None, :] >= sub_trajs_len[:, None]

            outs = self.model(sub_trajs_coor, src_padding_mask, sub_trajs_len)
            traj_embs.append(outs)

        traj_embs = torch.cat(traj_embs)
        pred_l1_simi = torch.cdist(traj_embs, traj_embs, 1)
        truth_l1_simi = datasets_simi
        pred_l1_simi_seq = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
        truth_l1_simi_seq = truth_l1_simi[torch.triu(torch.ones(truth_l1_simi.shape), diagonal = 1) == 1]

        loss = self.criterion(pred_l1_simi_seq, truth_l1_simi_seq)

        hrA = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 10, 10)
        hrB = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 50, 50)
        hrBinA = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 50, 10)

        return loss.item(), hrA, hrB, hrBinA
        

    def trajsimi_dataset_generator_batchi(self, trajs_coor):
        cur_index = 0
        len_datasets = len(trajs_coor)
        
        while cur_index < len_datasets:
            end_index = cur_index + mlp_batch_size \
                                if cur_index + mlp_batch_size < len_datasets \
                                else len_datasets

            sub_trajs_len = torch.tensor([len(trajs_coor[d_idx]) for d_idx in range(cur_index, end_index)], device = self.device, dtype = torch.long)

            sub_trajs_coor = [torch.tensor(trajs_coor[d_idx], dtype = torch.float) for d_idx in range(cur_index, end_index)]
            sub_trajs_coor = pad_sequence(sub_trajs_coor, batch_first = False) # should be sorted

            yield sub_trajs_coor.to(self.device), sub_trajs_len
            cur_index = end_index


    def trajsimi_dataset_generator_pairs_batchi(self, trajs_coor, datasets_simi, max_distance):
        len_datasets = len(trajs_coor)
        datasets_simi = torch.tensor(datasets_simi, device = self.device, dtype = torch.float)
        datasets_simi = (datasets_simi + datasets_simi.T) / max_distance
        
        count_i = 0
        batch_size = len_datasets if len_datasets < mlp_batch_size else mlp_batch_size
        counts = math.ceil( (len_datasets / batch_size)**2 )

        while count_i < counts:
            dataset_idxs_sample = random.sample(range(len_datasets), k = batch_size)
            dataset_idxs_sample.sort(key = lambda idx: len(trajs_coor[idx]), reverse = True) # len descending order

            sub_trajs_len = torch.tensor([len(trajs_coor[d_idx]) for d_idx in dataset_idxs_sample], device = self.device, dtype = torch.long)

            sub_trajs_coor = [torch.tensor(trajs_coor[d_idx], dtype = torch.float) for d_idx in dataset_idxs_sample]
            sub_trajs_coor = pad_sequence(sub_trajs_coor, batch_first = False) # should be sorted

            sub_simi = datasets_simi[dataset_idxs_sample][:,dataset_idxs_sample]

            yield sub_trajs_coor.to(self.device), sub_trajs_len, sub_simi
            count_i += 1


    def load_trajsimi_dataset(self):
        # 1. read TrajSimi dataset
        with open(Config.dataset_trajsimi_traj, 'rb') as fh:
            dic_dataset = pickle.load(fh)
            trajs_merc = dic_dataset['trajs_merc']
            
            
        with open(Config.dataset_trajsimi_dict, 'rb') as fh:
            dic_dataset = pickle.load(fh)
            train_simis = dic_dataset['train_simis']
            eval_simis = dic_dataset['eval_simis']
            test_simis = dic_dataset['test_simis']
            max_distance = dic_dataset['max_distance']

        trains_trajcoor = self.preprocessing_trajcoor(trajs_merc[:7000])
        evals_trajcoor = self.preprocessing_trajcoor(trajs_merc[7000:8000])
        tests_trajcoor = self.preprocessing_trajcoor(trajs_merc[8000:])

        logging.info("trajsimi dataset sizes. (trains/evals/tests={}/{}/{})" \
                    .format(len(trains_trajcoor), len(evals_trajcoor), len(tests_trajcoor)))

        return {'trains_trajcoor': trains_trajcoor, 'evals_trajcoor': evals_trajcoor, 'tests_trajcoor': tests_trajcoor, \
                'trains_simi': train_simis, 'evals_simi': eval_simis, 'tests_simi': test_simis, \
                'max_distance': max_distance}


    def preprocessing_trajcoor(self, lst_trajs: list):
        # lst_trajs = [ [[lon, lat_in_merc_space] ,[] ], ..., [], ..., ] 

        logging.info("trajcoor_to_trajgrid starts. #={}".format(len(lst_trajs)))
        _time = time.time()

        # local normlization
        xs, ys = zip(*[[p[0], p[1]] for traj in lst_trajs for p in traj])
        meanx, meany, stdx, stdy = np.mean(xs), np.mean(ys), np.std(xs), np.std(ys)
        lst_trajs_nodes_lonlat = [[[(p[0] - meanx) / stdx, (p[1] - meany) / stdy] for p in traj] for traj in lst_trajs]

        logging.info("trajcoor_to_trajgrid ends. @={:.3f}, #={}" \
                    .format(time.time() - _time, len(lst_trajs_nodes_lonlat)))
        # lst_trajs_nodes_lonlat = [ [[lon, lat_normalized] ,[] ], ..., [], ..., ] 
        return lst_trajs_nodes_lonlat


def parse_args():
    parser = argparse.ArgumentParser(description = "...")
    # dont give default value here! Otherwise, it will faultly overwrite the value in config.py.
    # config.py is the correct place to provide default values
    parser.add_argument('--debug', dest = 'debug', action='store_true')
    parser.add_argument('--dumpfile_uniqueid', type = str, help = '') # see config.py
    parser.add_argument('--seed', type = int, help = '')
    parser.add_argument('--dataset', type = str, help = '')
    parser.add_argument('--trajsimi_measure', type = str, help = '')
    

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))



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

    mlp = MLPTrainer()
    mlp.train()



