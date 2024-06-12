import sys
sys.path.append('..')
sys.path.append('../..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
import math
import random
import pickle
import argparse
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

from config import Config as Config
from utilities import tool_funcs
from nn.RSTS import Encoder 
from nn.RSTS_utils import load_rsts_region, Region

rsts_epochs = 5
rsts_batch_size = 128
rsts_learning_rate = 0.0005
rsts_training_bad_patience = 5


class TrajSimiRegression(nn.Module):
    def __init__(self, nin):
        # nin = traj_emb_size 
        super(TrajSimiRegression, self).__init__()
        self.enc = nn.Sequential(nn.Linear(nin, nin),
                                nn.ReLU(),
                                nn.Linear(nin, nin))

    def forward(self, trajs):
        # trajs: [batch_size, emb_size]
        return F.normalize(self.enc(trajs), dim=1) #[batch_size, emb_size]


class RSTSTrainer:
    def __init__(self):
        super(RSTSTrainer, self).__init__()
        self.device = torch.device('cuda:0')
        
        region_file = '{}/exp/snapshot/{}_RSTS_region.pkl'.format(Config.root_dir, Config.dataset_prefix)
        self.rsts_region = load_rsts_region(region_file)
        self.model = Encoder(Config.cell_embedding_dim, Config.traj_embedding_dim, self.rsts_region.vocal_nums, 
                        Config.rsts_num_layers, Config.rsts_dropout, Config.rsts_bidirectional, self.rsts_region)
        self.model.to(self.device)

        self.dic_datasets = self.load_trajsimi_dataset()
        
        self.checkpoint_path = '{}/{}_trajsimi_RSTS_{}_best{}.pt'.format(Config.snapshot_dir, \
                                    Config.dataset_prefix, Config.trajsimi_measure, Config.dumpfile_uniqueid)


    def train(self):
        logging.info("training. START! @={:.3f}".format(time.time()))
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        torch.autograd.set_detect_anomaly(True)
        
        cp = torch.load('{}/{}_RSTS_best_model.pt'.format(Config.snapshot_dir, Config.dataset_prefix))
        self.model.load_state_dict(cp['encoder'])
        self.model.to(self.device)
        
        self.regression = TrajSimiRegression(Config.traj_embedding_dim)
        self.regression.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.criterion.to(self.device)
        optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.model.parameters()))+
                                    list(filter(lambda p: p.requires_grad, self.regression.parameters())), \
                                    lr = rsts_learning_rate)

        best_hr_eval = -999999
        best_loss_train = 10000000.0
        best_epoch = 0
        bad_counter = 0
        bad_patience = rsts_training_bad_patience

        for i_ep in range(rsts_epochs):
            _time_ep = time.time()
            train_losses = []
            train_gpu = []
            train_ram = []

            self.model.train()
            self.regression.train()

            for i_batch, batch in enumerate( self.trajsimi_dataset_generator_pairs_batchi( \
                                                            self.dic_datasets['trains_cell'], \
                                                            self.dic_datasets['trains_simi'], \
                                                            self.dic_datasets['max_distance'])):
                _time_batch = time.time()
                optimizer.zero_grad()

                sub_trajs_cell, sub_trajs_len, sub_simi = batch
                # sub_trajs_img = input_processing(sub_trajs_merc, self.model.lon_range, self.model.lat_range, 
                #                                 self.model.imgsize_x_lr, self.model.imgsize_y_lr,
                #                                 self.model.pixelrange_lr).to(self.device)
                outs = self.model(sub_trajs_cell, sub_trajs_len)[1]
                outs = outs[sub_trajs_len-1, range(len(sub_trajs_len)), :]
                outs = self.regression(outs)
                pred_l1_simi = torch.cdist(outs, outs, 1) # use l1 here.
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
            eval_metrics = self.__test(self.dic_datasets['evals_cell'], \
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
                torch.save({'model': self.model.state_dict(), 'regression': self.regression.state_dict()}, 
                           self.checkpoint_path)
            else:
                bad_counter += 1

            if bad_counter == bad_patience or i_ep + 1 == rsts_epochs:
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
        self.regression.load_state_dict(checkpoint['regression'])
        self.regression.to(self.device)
        test_starttime = time.time()
        test_metrics = self.__test(self.dic_datasets['tests_cell'], \
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
    def __test(self, trajs_cell, datasets_simi, max_distance):
        self.model.eval()
        self.regression.eval()
        
        traj_embs = []
        datasets_simi = torch.tensor(datasets_simi, device = self.device, dtype = torch.float) / max_distance

        for i_batch, batch in enumerate(self.trajsimi_dataset_generator_batchi(trajs_cell)):
            sub_trajs_cell, sub_trajs_len = batch
            outs = self.model(sub_trajs_cell, sub_trajs_len)[1]
            outs = outs[sub_trajs_len-1, range(len(sub_trajs_len)), :]
            outs = self.regression(outs)
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
        

    def trajsimi_dataset_generator_batchi(self, trajs_cell):
        cur_index = 0
        len_datasets = len(trajs_cell)
        
        while cur_index < len_datasets:
            end_index = cur_index + rsts_batch_size \
                                if cur_index + rsts_batch_size < len_datasets \
                                else len_datasets
                                
            sub_trajs_cell = trajs_cell[cur_index: end_index]
            sub_traj_len = torch.tensor([len(traj) for traj in sub_trajs_cell], dtype = torch.long).to(self.device)
            sub_trajs_cell = pad_sequence(sub_trajs_cell, batch_first = False, padding_value = 0).to(self.device)
            yield sub_trajs_cell, sub_traj_len
            cur_index = end_index


    def trajsimi_dataset_generator_pairs_batchi(self, trajs_cell, datasets_simi, max_distance):
        len_datasets = len(trajs_cell)
        datasets_simi = torch.tensor(datasets_simi, device = self.device, dtype = torch.float) / max_distance
        
        count_i = 0
        batch_size = len_datasets if len_datasets < rsts_batch_size else rsts_batch_size
        counts = math.ceil( (len_datasets / batch_size)**2 )

        while count_i < counts:
            dataset_idxs_sample = random.sample(range(len_datasets), k = batch_size)
            sub_trajs_cell = [trajs_cell[idx] for idx in dataset_idxs_sample]
            sub_traj_len = torch.tensor([len(traj) for traj in sub_trajs_cell], dtype = torch.long).to(self.device)
            
            sub_trajs_cell = pad_sequence(sub_trajs_cell, batch_first = False, padding_value = 0).to(self.device)
            sub_simi = datasets_simi[dataset_idxs_sample][:,dataset_idxs_sample]

            yield sub_trajs_cell, sub_traj_len, sub_simi
            count_i += 1


    def load_trajsimi_dataset(self):
        # 1. read TrajSimi dataset
        # 2. convert merc_seq to cell_id_seqs
        
        with open(Config.dataset_trajsimi_traj, 'rb') as fh:
            dic_dataset = pickle.load(fh)
            trajs_merc = dic_dataset['trajs_merc']
            trajs_ts = dic_dataset['trajs_ts']
            
            
        with open(Config.dataset_trajsimi_dict, 'rb') as fh:
            dic_dataset = pickle.load(fh)
            train_simis = dic_dataset['train_simis']
            eval_simis = dic_dataset['eval_simis']
            test_simis = dic_dataset['test_simis']
            max_distance = dic_dataset['max_distance']
            
            trains_merc = trajs_merc[:7000]
            evals_merc = trajs_merc[7000:8000]
            tests_merc = trajs_merc[8000:]
            
            trains_ts = trajs_ts[:7000]
            evals_ts = trajs_ts[7000:8000]
            tests_ts = trajs_ts[8000:]
            
            trains_cell = self.trajs_merc_to_cells(trains_merc, trains_ts)
            evals_cell = self.trajs_merc_to_cells(evals_merc, evals_ts)
            tests_cell = self.trajs_merc_to_cells(tests_merc, tests_ts)

        logging.info("trajsimi dataset sizes. (trains/evals/tests={}/{}/{})" \
                    .format(len(trains_merc), len(evals_merc), len(tests_merc)))

        return {'trains_cell': trains_cell, 'evals_cell': evals_cell, 'tests_cell': tests_cell, \
                'trains_simi': train_simis, 'evals_simi': eval_simis, 'tests_simi': test_simis, \
                'max_distance': max_distance}


    def trajs_merc_to_cells(self, trajs_merc, trajs_ts):
        trajs_cellid = []
        for i in range(len(trajs_merc)):
            xy = trajs_merc[i]
            t = trajs_ts[i]
            trajs_cellid.append(torch.tensor(self.rsts_region.trip2words(xy, t)))
        return trajs_cellid


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


# nohup python RSTS_trajsimi_train.py --dataset xian --trajsimi_measure stedr --seed 2000 --debug &> ../result &
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

    rsts = RSTSTrainer()
    rsts.train()



