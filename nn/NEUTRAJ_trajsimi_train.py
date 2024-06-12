import sys
sys.path.append('..')
sys.path.append('../..')
import os
import math
import logging
import time
import pickle
import torch
import numpy as np
import argparse

from config import Config as Config
from utilities import tool_funcs
from nn.NEUTRAJ_core import NeuTraj_Network, WeightedRankingLoss
from nn.NEUTRAJ_utils import neutraj_trajs_preprocess, neutraj_trajs_process_for_model_input
from nn.NEUTRAJ_utils import distance_sampling, negative_distance_sampling
from nn.NEUTRAJ_utils import trajcoor_to_trajpadinput


neutraj_cell_size = 100
neutraj_epochs = 20 
neutraj_batch_size = 128
neutraj_learning_rate = 0.001
neutraj_training_bad_patience = 20
neutraj_sam_spatial_width = 2
neutraj_sampling_num = 10

class NEUTRAJTrainer:
    def __init__(self):
        super(NEUTRAJTrainer, self).__init__()
        self.device = torch.device("cuda:0")
        
        x_min, y_min = tool_funcs.lonlat2meters(Config.min_lon, Config.min_lat)
        x_max, y_max = tool_funcs.lonlat2meters(Config.max_lon, Config.max_lat)
        self.x_range = (x_min, x_max)
        self.y_range = (y_min, y_max)
        self.grid_size = int(math.ceil(max(x_max - x_min, y_max - y_min) / neutraj_cell_size) \
                            + neutraj_sam_spatial_width * 2 + 2)
        logging.debug('NeuTraj_Network.grid_cells={}*{}'.format(self.grid_size, self.grid_size))

        self.dic_datasets = self.load_trajsimi_dataset()
        
        self.checkpoint_path = '{}/{}_trajsimi_NEUTRAJ_{}_best{}.pt'.format( \
                                    Config.snapshot_dir, Config.dataset_prefix, \
                                    Config.trajsimi_measure, Config.dumpfile_uniqueid)
        
    def train(self):
        logging.info("training. START! @={:.3f}".format(time.time()))
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        torch.autograd.set_detect_anomaly(True)
        in_cell_update = True

        self.model = NeuTraj_Network(4, Config.traj_embedding_dim, [self.grid_size, self.grid_size], \
                                    neutraj_cell_size, self.x_range, self.y_range)
        self.model.to(self.device)

        self.criterion = WeightedRankingLoss(neutraj_batch_size, neutraj_sampling_num)
        self.criterion.to(self.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), \
                                            lr = neutraj_learning_rate)

        best_hr_eval = 0.0
        best_loss_train = 10000000.0
        best_epoch = 0
        bad_counter = 0
        bad_patience = neutraj_training_bad_patience
        timetoreport = [1200, 2400, 3600] # len may change later

        for i_ep in range(neutraj_epochs):
            _time_ep = time.time()
            train_losses = []

            self.model.train()

            for i_batch, batch in enumerate( self.trajsimi_dataset_generator_batchi( \
                                                            self.dic_datasets['trains_trajpad'], \
                                                            self.dic_datasets['trains_trajlen'], \
                                                            self.dic_datasets['trains_simi'], \
                                                            self.dic_datasets['max_distance'])):
                _time_batch = time.time()
                
                # inputs_arrays = [[ [-1.1607499926658438, 1.299966075551713, 34.0, 202.0] ]]
                inputs_arrays, inputs_len_arrays, distance_arrays = batch[0], batch[1], batch[2]

                trajs_loss, negative_loss = self.model(inputs_arrays, inputs_len_arrays)

                positive_distance_target = torch.Tensor(distance_arrays[0]).view((-1, 1)).to(self.device)
                negative_distance_target = torch.Tensor(distance_arrays[1]).view((-1, 1)).to(self.device)

                loss = self.criterion(trajs_loss, positive_distance_target,
                                    negative_loss, negative_distance_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if not in_cell_update:
                    self.model.spatial_memory_update(inputs_arrays, inputs_len_arrays)

                train_losses.append(loss.item())

                # debug output
                if i_batch % 10 == 0 and Config.debug:
                    logging.debug("training. ep-batch={}-{}, train_loss={:.4f}, @={:.3f}" \
                                .format(i_ep, i_batch, loss.item(), time.time()-_time_batch,))
                
                # exp of training time vs. effectiveness
                if Config.trajsimi_timereport_exp and len(timetoreport) \
                        and time.time() - training_starttime >= timetoreport[0]:
                    test_metrics = self.__test(self.dic_datasets['tests_trajpad'], \
                                                self.dic_datasets['tests_trajlen'], \
                                                self.dic_datasets['tests_simi'], \
                                                self.dic_datasets['max_distance'])
                    logging.info("test. ts={}, hr={:.3f},{:.3f},{:.3f}".format(timetoreport[0], *test_metrics))
                    timetoreport.pop(0)
                    self.model.train()

            # ep debug output
            logging.info("training. i_ep={}, loss={:.4f}, @={:.3f}" \
                        .format(i_ep, tool_funcs.mean(train_losses), time.time()-_time_ep))
            
            # eval
            eval_metrics = self.__test(self.dic_datasets['evals_trajpad'], \
                                        self.dic_datasets['evals_trajlen'], \
                                        self.dic_datasets['evals_simi'], \
                                        self.dic_datasets['max_distance'])
            logging.info("eval.     i_ep={}, hr={:.3f},{:.3f},{:.3f}".format(i_ep, *eval_metrics))
            eval_hr_ep = eval_metrics[0]

            # early stopping
            if  eval_hr_ep > best_hr_eval:
                best_epoch = i_ep
                best_hr_eval = eval_hr_ep
                best_loss_train = train_losses[-1]
                bad_counter = 0
                torch.save({'model': self.model.state_dict()}, self.checkpoint_path)
            else:
                bad_counter += 1

            if bad_counter == bad_patience or i_ep + 1 == neutraj_epochs:
                training_endtime = time.time()
                logging.info("training end. @={:.0f}, best_epoch={}, best_loss_train={:.4f}, best_hr_eval={:.4f}, #param={}" \
                            .format(training_endtime - training_starttime, \
                                    best_epoch, best_loss_train, best_hr_eval, \
                                    tool_funcs.num_of_model_params(self.model)))
                break
            
        # test
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        
        test_metrics = self.__test(self.dic_datasets['tests_trajpad'], \
                                    self.dic_datasets['tests_trajlen'], \
                                    self.dic_datasets['tests_simi'], \
                                    self.dic_datasets['max_distance'])
        logging.info("test.     hr={:.3f},{:.3f},{:.3f}".format(*test_metrics))


    # tranplant from test_methods.test_model() and re-implement......
    # for inner calling only, can also be used for testing
    @torch.no_grad()
    def __test(self, trajs_pad_input, trajs_length, datasets_simi, max_distance):
        self.model.eval()
        
        # trajs_pad_input, trajs_length = neutraj_trajs_process_for_model_input(datasets_coor, datasets_gridxy, max_traj_len)
        embs = self.__get_embeddings(trajs_pad_input, trajs_length)

        # preds = torch.exp(-torch.square(torch.cdist(embs, embs, 2)))
        # preds = preds[torch.triu(torch.ones(preds.shape), diagonal = 1) == 1]
        # gtruth = torch.tensor(datasets_simi, dtype = torch.float, device = Config.device)
        # gtruth = gtruth[torch.triu(torch.ones(gtruth.shape), diagonal = 1) == 1]
        preds = torch.square(torch.cdist(embs, embs, 2))
        gtruth = torch.tensor(datasets_simi, dtype = torch.float, device = self.device)

        # metrics
        hrA = tool_funcs.hitting_ratio(preds, gtruth, 10, 10)
        hrB = tool_funcs.hitting_ratio(preds, gtruth, 50, 50)
        hrBinA = tool_funcs.hitting_ratio(preds, gtruth, 50, 10)

        return hrA, hrB, hrBinA


    @torch.no_grad()
    def __get_embeddings(self, trajs_pad_input, trajs_length):
        self.model.eval()
        
        embeddings_list = []
        cur_index = 0
        len_datasets = len(trajs_pad_input)

        while cur_index < len_datasets:
            end_index = cur_index + neutraj_batch_size if cur_index + neutraj_batch_size < len_datasets else len_datasets

            hidden = (torch.zeros((end_index-cur_index, Config.traj_embedding_dim), requires_grad=False, dtype = torch.float, device=self.device),
                        torch.zeros((end_index-cur_index, Config.traj_embedding_dim), requires_grad=False, dtype = torch.float, device=self.device))
            out = self.model.rnn([torch.tensor(trajs_pad_input[cur_index: end_index], requires_grad=False, dtype = torch.float, device=self.device), \
                                   trajs_length[cur_index: end_index]], hidden)
            
            embeddings_list.append(out.data)
            cur_index = end_index
        embeddings_list = torch.cat(embeddings_list, dim=0)
        # return embeddings_list.cpu().numpy()
        return embeddings_list

    
    # transplant from neutraj_trainer.batch_generator()
    # original implementation has too many loops, very very very slow, but i amnot willing to optimize it...
    def trajsimi_dataset_generator_batchi(self, datasets_input, datasets_input_reallen, datasets_simi, max_distance):
        datasets_simi = np.array(datasets_simi) / max_distance
        
        len_datasets = len(datasets_input)
        # datasets_input_t = torch.tensor(datasets_input, dtype = torch.float, device = Config.device)
        cur_index = 0
        while cur_index < len_datasets:
            end_index = cur_index + neutraj_batch_size if cur_index + neutraj_batch_size < len_datasets else len_datasets
            
            # # assume end_index - cur_index = 128
            # anchor_input = datasets_input_t[list(range(cur_index, end_index))]
            # anchor_input = torch.repeat_interleave(anchor_input, 1 + Config.neutraj_sampling_num, dim = 0) # size = [1408, 50, 4]; tensor
            
            # anchor_input_len = [datasets_input_reallen[idx] for idx in range(cur_index, end_index)] # size: [128]
            # anchor_input_len = sum([ [v] * (1 + Config.neutraj_sampling_num) for v in anchor_input_len], []) # size: [1408]; list

            # trajs_input, negative_input = [], []
            # trajs_input_len, negative_input_len = [], []
            # distance, negative_distance = [], []
            # batch_trajs_keys = {}
            # batch_trajs_input, batch_trajs_len = [], []

            # for _i in range(cur_index, end_index):
            #     sampling_index_list = distance_sampling(datasets_simi, len_datasets, _i) # list
            #     negative_sampling_index_list = negative_distance_sampling(datasets_simi, len_datasets, _i) # list
                
            #     trajs_input.append(datasets_input_t[[_i] + sampling_index_list])
            #     trajs_input_len.extend( [ datasets_input_reallen[idx] for idx in ([_i] + sampling_index_list) ] )

            #     negative_input.append(datasets_input_t[[_i] + negative_sampling_index_list])
            #     negative_input_len.extend( [ datasets_input_reallen[idx] for idx in ([_i] + negative_sampling_index_list) ] )

            #     distance.extend( [1] \
            #                     + [ np.exp(-float(datasets_simi[_i][idx])*Config.neutraj_sampling_num) for idx in sampling_index_list ])
            #     negative_distance.extend( [1] \
            #                     + [ np.exp(-float(datasets_simi[_i][idx])*Config.neutraj_sampling_num) for idx in negative_sampling_index_list ])

            #     if _i not in batch_trajs_keys:
            #         batch_trajs_keys[_i] = 0
            #         batch_trajs_input.append(datasets_input_t[_i])
            #         batch_trajs_len.append(datasets_input_reallen[_i])

            #     for idx in sampling_index_list:
            #         if idx not in batch_trajs_keys:
            #             batch_trajs_keys[_i] = 0
            #             batch_trajs_input.append(datasets_input_t[idx])
            #             batch_trajs_len.append(datasets_input_reallen[idx])
                
            #     for idx in negative_sampling_index_list:
            #         if idx not in batch_trajs_keys:
            #             batch_trajs_keys[_i] = 0
            #             batch_trajs_input.append(datasets_input_t[idx])
            #             batch_trajs_len.append(datasets_input_reallen[idx])

            # trajs_input = torch.cat(trajs_input, dim = 0) # size = [1408, 50, 4]; tensor
            # negative_input = torch.cat(negative_input, dim = 0) # size = [1408, 50, 4]; tensor
            # batch_trajs_input = torch.stack(batch_trajs_input) # size = [?, 50, 4]; tensor

            # yield ([anchor_input, trajs_input, negative_input, batch_trajs_input],
            #        [anchor_input_len, trajs_input_len, negative_input_len, batch_trajs_len],
            #        [np.array(distance),np.array(negative_distance)])


            
            anchor_input, trajs_input, negative_input, distance, negative_distance = [], [], [], [], []
            anchor_input_len, trajs_input_len, negative_input_len = [], [], []
            batch_trajs_keys = {}
            batch_trajs_input, batch_trajs_len = [], []

            for _i in range(cur_index, end_index):

                sampling_index_list = distance_sampling(datasets_simi, len_datasets, _i, neutraj_sampling_num)
                negative_sampling_index_list = negative_distance_sampling(datasets_simi, len_datasets, _i, neutraj_sampling_num)

                trajs_input.append(datasets_input[_i]) #
                anchor_input.append(datasets_input[_i])
                negative_input.append(datasets_input[_i])
                if _i not in batch_trajs_keys:
                    batch_trajs_keys[_i] = 0
                    batch_trajs_input.append(datasets_input[_i])
                    batch_trajs_len.append(datasets_input_reallen[_i])

                anchor_input_len.append(datasets_input_reallen[_i])
                trajs_input_len.append(datasets_input_reallen[_i])
                negative_input_len.append(datasets_input_reallen[_i])

                distance.append(1)
                negative_distance.append(1)

                for traj_index in sampling_index_list:

                    anchor_input.append(datasets_input[_i])
                    trajs_input.append(datasets_input[traj_index])

                    anchor_input_len.append(datasets_input_reallen[_i])
                    trajs_input_len.append(datasets_input_reallen[traj_index])

                    if traj_index not in batch_trajs_keys:
                        batch_trajs_keys[_i] = 0
                        batch_trajs_input.append(datasets_input[traj_index])
                        batch_trajs_len.append(datasets_input_reallen[traj_index])

                    distance.append(np.exp(-float(datasets_simi[_i][traj_index])*neutraj_sampling_num))

                for traj_index in negative_sampling_index_list:

                    negative_input.append(datasets_input[traj_index])
                    negative_input_len.append(datasets_input_reallen[traj_index])
                    negative_distance.append(np.exp(-float(datasets_simi[_i][traj_index])*neutraj_sampling_num))

                    if traj_index not in batch_trajs_keys:
                        batch_trajs_keys[_i] = 0
                        batch_trajs_input.append(datasets_input[traj_index])
                        batch_trajs_len.append(datasets_input_reallen[traj_index])

            # max_anchor_length = max(anchor_input_len)
            # max_sample_lenght = max(trajs_input_len)
            # max_neg_lenght = max(negative_input_len)
            # anchor_input = pad_sequence(anchor_input, maxlen=max_anchor_length)
            # trajs_input = pad_sequence(trajs_input, maxlen=max_sample_lenght)
            # negative_input = pad_sequence(negative_input, maxlen=max_neg_lenght)
            # batch_trajs_input = pad_sequence(batch_trajs_input, maxlen=max(max_anchor_length, max_sample_lenght,
                                                                        #    max_neg_lenght))
            yield ([np.array(anchor_input),np.array(trajs_input),np.array(negative_input), np.array(batch_trajs_input)],
                   [anchor_input_len, trajs_input_len, negative_input_len, batch_trajs_len],
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
            
            trains_merc = trajs_merc[:7000]
            evals_merc = trajs_merc[7000:8000]
            tests_merc = trajs_merc[8000:]

            trains_trajpad, trains_trajlen = trajcoor_to_trajpadinput(trains_merc, self.x_range, self.y_range, neutraj_cell_size)
            evals_trajpad, evals_trajlen = trajcoor_to_trajpadinput(evals_merc, self.x_range, self.y_range, neutraj_cell_size)
            tests_trajpad, tests_trajlen = trajcoor_to_trajpadinput(tests_merc, self.x_range, self.y_range, neutraj_cell_size)



        return {'trains_trajpad': trains_trajpad, 'evals_trajpad': evals_trajpad, 'tests_trajpad': tests_trajpad, \
                'trains_trajlen': trains_trajlen, 'evals_trajlen': evals_trajlen, 'tests_trajlen': tests_trajlen, \
                'trains_simi': train_simis, 'evals_simi': eval_simis, 'tests_simi': test_simis, \
                'max_distance': max_distance}


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


# nohup python NEUTRAJ_trajsimi_train.py --dataset xian --trajsimi_measure dtw --seed 2000 --debug &> ../result &

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
    neutraj = NEUTRAJTrainer()
    metrics = neutraj.train()



