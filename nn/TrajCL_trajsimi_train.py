import os
import sys
sys.path.append('..')
sys.path.append('../..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import time
import logging
import math
import random
import pickle
import argparse
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from functools import partial

from config import Config as Config
from nn.utils.cellspace import CellSpace
from nn.TrajCL import TrajCLMoCo, TrajCL
from nn.utils.traj import get_aug_fn
from nn.node2vec_ import train_node2vec
from nn.utils.traj import merc2cell2, generate_spatial_features
from utilities import tool_funcs
from utilities.tool_funcs import hitting_ratio

trajcl_cell_sidelen = 100
trajcl_cellspace_buffer = 500

trajcl_cl_epochs = 50 
trajcl_cl_batch_size = 128
trajcl_cl_learning_rate = 0.001
trajcl_cl_training_bad_patience = 5
trajcl_cl_training_lr_degrade_step = 5
trajcl_cl_training_lr_degrade_gamma = 0.5

trajcl_trajsimi_epochs = 20
trajcl_trajsimi_batch_size = 256
trajcl_trajsimi_learning_rate = 0.001
trajcl_trajsimi_training_bad_patience = 5



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


class TrajCLTrainer:
    def __init__(self):
        super(TrajCLTrainer, self).__init__()
        self.device = torch.device('cuda:0')
        
        self.aug1 = get_aug_fn("mask")
        self.aug2 = get_aug_fn("subset")

        x_min, y_min = tool_funcs.lonlat2meters(Config.min_lon, Config.min_lat)
        x_max, y_max = tool_funcs.lonlat2meters(Config.max_lon, Config.max_lat)
        x_min -= trajcl_cellspace_buffer
        y_min -= trajcl_cellspace_buffer
        x_max += trajcl_cellspace_buffer
        y_max += trajcl_cellspace_buffer

        self.cellspace = CellSpace(trajcl_cell_sidelen, trajcl_cell_sidelen, 
                                    x_min, y_min, x_max, y_max)
        
        self.clmodel = None
        self.model = None # trajcl
        
        self.checkpoint_path_cl = '{}/{}_trajsimi_TrajCL_{}_best{}.pt'.format(Config.snapshot_dir, \
                                    Config.dataset_prefix, Config.cell_embedding_dim, Config.dumpfile_uniqueid)
        self.checkpoint_path_trajsimi = '{}/{}_trajsimi_TrajCL_{}_{}_best{}.pt'.format(Config.snapshot_dir, \
                                    Config.dataset_prefix, Config.trajsimi_measure, 
                                    Config.cell_embedding_dim, Config.dumpfile_uniqueid)


    def _cell_emb_pretrain(self):
        device = torch.device('cuda:0')
        _, edge_index = self.cellspace.all_neighbour_cell_pairs_permutated_optmized()
        edge_index = torch.tensor(edge_index, dtype = torch.long, device = device).T
        self.embs = train_node2vec(edge_index, Config.cell_embedding_dim, device) # tensor
    
    
    def cl_pretrain(self):
        encoder_q = TrajCL(self.cellspace, Config.cell_embedding_dim).to(self.device)
        encoder_k = TrajCL(self.cellspace, Config.cell_embedding_dim).to(self.device)
        self.clmodel = TrajCLMoCo(encoder_q, encoder_k).to(self.device)
        
        if os.path.exists(self.checkpoint_path_cl):
            logging.info('Skip cl_pretrain. Load from cp file.')
            self.load_checkpoint_cl()
            return 
        
        self._cell_emb_pretrain()
        
        dic_datasets = self.load_trajsimi_dataset()
        trains_dataset = TrajDataset(dic_datasets['trains_merc'])
        train_dataloader = DataLoader(trains_dataset, 
                                            batch_size = trajcl_cl_batch_size, 
                                            shuffle = False, 
                                            num_workers = 0, 
                                            drop_last = True, 
                                            collate_fn = partial(collate_and_augment, 
                                                                cellspace = self.cellspace, 
                                                                embs = self.embs, 
                                                                augfn1 = self.aug1, augfn2 = self.aug2, 
                                                                device = self.device) )

        optimizer = torch.optim.Adam(self.clmodel.parameters(), lr = trajcl_cl_learning_rate, weight_decay = 0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = trajcl_cl_training_lr_degrade_step, gamma = trajcl_cl_training_lr_degrade_gamma)

        training_starttime = time.time()
        logging.info("[Training] START! timestamp={:.0f}".format(training_starttime))
        torch.autograd.set_detect_anomaly(True)

        best_loss_train = 100000
        best_epoch = 0
        bad_counter = 0
        bad_patience = trajcl_cl_training_bad_patience

        for i_ep in range(trajcl_cl_epochs):
            _time_ep = time.time()
            loss_ep = []

            self.clmodel.train()

            for i_batch, batch in enumerate(train_dataloader):
                _time_batch = time.time()
                optimizer.zero_grad()

                trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len = batch

                model_rtn = self.clmodel(trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len)
                loss = self.clmodel.loss(*model_rtn)

                loss.backward()
                optimizer.step()
                loss_ep.append(loss.item())

                if i_batch % 10 == 0:
                    logging.debug("[Training] ep-batch={}-{}, loss={:.3f}, @={:.3f}" \
                            .format(i_ep, i_batch, loss.item(), time.time() - _time_batch))

            scheduler.step() # decay before optimizer when pytorch < 1.1
              
            loss_ep_avg = tool_funcs.mean(loss_ep)
            logging.info("[Training] ep={}: avg_loss={:.3f}, @={:.3f}" \
                    .format(i_ep, loss_ep_avg, time.time() - _time_ep))
            
            # early stopping 
            if loss_ep_avg < best_loss_train:
                best_epoch = i_ep
                best_loss_train = loss_ep_avg
                bad_counter = 0
                self.save_checkpoint_cl()
            else:
                bad_counter += 1

            if bad_counter == bad_patience or (i_ep + 1) == trajcl_cl_epochs:
                logging.info("[Training] END! @={}, best_epoch={}, best_loss_train={:.6f}" \
                            .format(time.time()-training_starttime, best_epoch, best_loss_train))
                break
        
        return
    
    
    def save_checkpoint_cl(self):
        torch.save({'encoder_q_state_dict': self.clmodel.clmodel.encoder_q.state_dict(),
                    'embs': self.embs,
                    'gamma_param': self.clmodel.clmodel.encoder_q.gamma_param,
                    'aug1': self.aug1.__name__,
                    'aug2': self.aug2.__name__},
                    self.checkpoint_path_cl)
        return
    
    
    def load_checkpoint_cl(self):
        cp = torch.load(self.checkpoint_path_cl)
        self.clmodel.clmodel.encoder_q.load_state_dict(cp['encoder_q_state_dict'])
        self.clmodel.clmodel.encoder_q.gamma_param = cp['gamma_param']
        self.embs = cp['embs'].to(self.device)
        self.clmodel.to(self.device)

    
    def train_trajsimi(self):
        logging.info("training. START! @={:.3f}".format(time.time()))
        training_starttime = time.time()
        torch.autograd.set_detect_anomaly(True)
        
        self.load_checkpoint_cl()
        self.model = self.clmodel.clmodel.encoder_q
        self.trajsimiregression = TrajSimiRegression(Config.traj_embedding_dim)
        self.trajsimiregression.to(self.device)

        self.criterion = nn.MSELoss()
        self.criterion.to(self.device)
        optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.model.parameters()))+
                                    list(filter(lambda p: p.requires_grad, self.trajsimiregression.parameters())), \
                                    lr = trajcl_trajsimi_learning_rate)

        dic_datasets = self.load_trajsimi_dataset()

        best_hr_eval = 0.0
        best_loss_train = 10000000.0
        best_epoch = 0
        bad_counter = 0
        bad_patience = trajcl_trajsimi_training_bad_patience
        timetoreport = [1200, 2400, 3600] # len may change later

        for i_ep in range(trajcl_trajsimi_epochs):
            _time_ep = time.time()
            train_losses = []

            self.model.train()
            self.trajsimiregression.train()

            for i_batch, batch in enumerate( self.trajsimi_dataset_generator_pairs_batchi( \
                                                            dic_datasets['trains_merc'], \
                                                            dic_datasets['trains_simi'], \
                                                            dic_datasets['max_distance'])):
                _time_batch = time.time()
                optimizer.zero_grad()

                trajs_emb, trajs_emb_p, trajs_len, sub_simi = batch
                max_trajs_len = trajs_len.max().item() 
                src_padding_mask = torch.arange(max_trajs_len, device = self.device)[None, :] >= trajs_len[:, None]
                embs = self.model(trajs_emb, None, src_padding_mask, trajs_len, trajs_emb_p)
                outs = self.trajsimiregression(embs)
                
                pred_l1_simi = torch.cdist(outs, outs, 1)
                pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
                truth_l1_simi = sub_simi[torch.triu(torch.ones(sub_simi.shape), diagonal = 1) == 1]
                loss_train = self.criterion(pred_l1_simi, truth_l1_simi)

                loss_train.backward()
                optimizer.step()
                train_losses.append(loss_train.item())

                if i_batch % 200 == 0:
                    logging.debug("training. ep-batch={}-{}, train_loss={:.4f}, @={:.3f}" \
                                .format(i_ep, i_batch, loss_train.item(), time.time()-_time_batch, ))

                # exp of training time vs. effectiveness
                if Config.trajsimi_timereport_exp and len(timetoreport) \
                            and time.time() - training_starttime >= timetoreport[0]:
  
                    test_metrics = self.test(dic_datasets['tests_merc'], dic_datasets['tests_simi'], dic_datasets['max_distance'])
                    logging.info("test.   ts={}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f}".format(timetoreport[0], *test_metrics))
                    timetoreport.pop(0)
                    self.model.train()
                    self.trajsimiregression.train()

            # i_ep
            logging.info("training. i_ep={}, loss={:.4f}, @={:.3f}" \
                        .format(i_ep, tool_funcs.mean(train_losses), time.time()-_time_ep))
            
            eval_metrics = self.test(dic_datasets['evals_merc'], dic_datasets['evals_simi'], dic_datasets['max_distance'])
            logging.info("eval.     i_ep={}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f}".format(i_ep, *eval_metrics))
            
            hr_eval_ep = eval_metrics[1]

            # early stopping
            if  hr_eval_ep > best_hr_eval:
                best_epoch = i_ep
                best_hr_eval = hr_eval_ep
                bad_counter = 0
                torch.save({"encoder_q" : self.model.state_dict(),
                            'gamma_param': self.model.gamma_param,
                            "trajsimi": self.trajsimiregression.state_dict()}, 
                            self.checkpoint_path_trajsimi)
                        
            else:
                bad_counter += 1

            if bad_counter == bad_patience or i_ep + 1 == trajcl_trajsimi_epochs:
                training_endtime = time.time()
                logging.info("training end. @={:.3f}, best_epoch={}, best_hr_eval={:.4f}" \
                            .format(training_endtime - training_starttime, best_epoch, best_hr_eval))
                break
            
        # test
        checkpoint = torch.load(self.checkpoint_path_trajsimi)
        self.model.load_state_dict(checkpoint['encoder_q'])
        self.model.to(self.device)
        self.model.gamma_param = checkpoint['gamma_param']
        self.trajsimiregression.load_state_dict(checkpoint['trajsimi'])
        self.trajsimiregression.to(self.device)
        
        test_metrics = self.test(dic_datasets['tests_merc'], dic_datasets['tests_simi'], dic_datasets['max_distance'])
        logging.info("test.     loss= {:.4f}, hr={:.3f},{:.3f},{:.3f}".format(*test_metrics))
        return     


    @torch.no_grad()
    def test(self, datasets, datasets_simi, max_distance):
        # prepare dataset
        self.model.eval()
        self.trajsimiregression.eval()

        datasets_simi = torch.tensor(datasets_simi, device = self.device, dtype = torch.float) / max_distance
        traj_outs = []

        # get traj embeddings 
        for i_batch, batch in enumerate(self.trajsimi_dataset_generator_batchi(datasets)):
            trajs_emb, trajs_emb_p, trajs_len = batch
            max_trajs_len = trajs_len.max().item() 
            src_padding_mask = torch.arange(max_trajs_len, device = self.device)[None, :] >= trajs_len[:, None]
            embs = self.model(trajs_emb, None, src_padding_mask, trajs_len, trajs_emb_p)
            outs = self.trajsimiregression(embs)
            traj_outs.append(outs)
        
        # calculate similarity
        traj_outs = torch.cat(traj_outs)
        pred_l1_simi = torch.cdist(traj_outs, traj_outs, 1)
        truth_l1_simi = datasets_simi
        pred_l1_simi_seq = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
        truth_l1_simi_seq = truth_l1_simi[torch.triu(torch.ones(truth_l1_simi.shape), diagonal = 1) == 1]

        # metrics
        loss = self.criterion(pred_l1_simi_seq, truth_l1_simi_seq)
        hrA = hitting_ratio(pred_l1_simi, truth_l1_simi, 10, 10)
        hrB = hitting_ratio(pred_l1_simi, truth_l1_simi, 50, 50)
        hrBinA = hitting_ratio(pred_l1_simi, truth_l1_simi, 50, 10)

        return loss.item(), hrA, hrB, hrBinA

        
    @torch.no_grad()
    def trajsimi_dataset_generator_batchi(self, datasets):
        cur_index = 0
        len_datasets = len(datasets)

        while cur_index < len_datasets:
            end_index = cur_index + trajcl_trajsimi_batch_size \
                                if cur_index + trajcl_trajsimi_batch_size < len_datasets \
                                else len_datasets

            trajs = [datasets[d_idx] for d_idx in range(cur_index, end_index)]

            trajs_cell, trajs_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs])
            trajs_emb_p = [torch.tensor(generate_spatial_features(t, self.cellspace)) for t in trajs_p]
            trajs_emb_p = pad_sequence(trajs_emb_p, batch_first = False).to(self.device)

            trajs_emb_cell = [self.embs[list(t)] for t in trajs_cell]
            trajs_emb_cell = pad_sequence(trajs_emb_cell, batch_first = False).to(self.device) # [seq_len, batch_size, emb_dim]
                
            trajs_len = torch.tensor(list(map(len, trajs_cell)), dtype = torch.long, device = self.device)
            yield trajs_emb_cell, trajs_emb_p, trajs_len

            cur_index = end_index


    def trajsimi_dataset_generator_pairs_batchi(self, trajs_merc, datasets_simi, max_distance):
        len_datasets = len(trajs_merc)
        datasets_simi = torch.tensor(datasets_simi, device = self.device, dtype = torch.float) / max_distance
        
        count_i = 0
        batch_size = len_datasets if len_datasets < trajcl_trajsimi_batch_size else trajcl_trajsimi_batch_size
        counts = math.ceil( (len_datasets / batch_size)**2 )

        while count_i < counts:
            dataset_idxs_sample = random.sample(range(len_datasets), k = batch_size)
            sub_simi = datasets_simi[dataset_idxs_sample][:,dataset_idxs_sample]

            trajs = [trajs_merc[d_idx] for d_idx in dataset_idxs_sample]
            trajs_cell, trajs_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs])
            trajs_emb_p = [torch.tensor(generate_spatial_features(t, self.cellspace)) for t in trajs_p]
            trajs_emb_p = pad_sequence(trajs_emb_p, batch_first = False).to(self.device)

            trajs_emb_cell = [self.embs[list(t)] for t in trajs_cell]
            trajs_emb_cell = pad_sequence(trajs_emb_cell, batch_first = False).to(self.device) # [seq_len, batch_size, emb_dim]
            
            trajs_len = torch.tensor(list(map(len, trajs_cell)), dtype = torch.long, device = self.device)

            yield trajs_emb_cell, trajs_emb_p, trajs_len, sub_simi
            count_i += 1


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


        return {'trains_merc': trains_merc, 'evals_merc': evals_merc, 'tests_merc': tests_merc, \
                'trains_simi': train_simis, 'evals_simi': eval_simis, 'tests_simi': test_simis, \
                'max_distance': max_distance}


class TrajDataset(Dataset):
    def __init__(self, data):
        # data: DataFrame
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_and_augment(trajs, cellspace, embs, augfn1, augfn2, device):
    # trajs: list of [[lon, lat], [,], ...]

    # 1. augment the input traj for forming 2 input views
    # 2. convert augmented trajs to trajs based on mercator space by cells
    # 3. read cell embeddings and form batch tensors (sort, pad)

    trajs1 = [augfn1(t) for t in trajs]
    trajs2 = [augfn2(t) for t in trajs]

    trajs1_cell, trajs1_p = zip(*[merc2cell2(t, cellspace) for t in trajs1])
    trajs2_cell, trajs2_p = zip(*[merc2cell2(t, cellspace) for t in trajs2])

    trajs1_emb_p = [torch.tensor(generate_spatial_features(t, cellspace)) for t in trajs1_p]
    trajs2_emb_p = [torch.tensor(generate_spatial_features(t, cellspace)) for t in trajs2_p]

    trajs1_emb_p = pad_sequence(trajs1_emb_p, batch_first = False).to(device)
    trajs2_emb_p = pad_sequence(trajs2_emb_p, batch_first = False).to(device)

    trajs1_emb_cell = [embs[list(t)] for t in trajs1_cell]
    trajs2_emb_cell = [embs[list(t)] for t in trajs2_cell]

    trajs1_emb_cell = pad_sequence(trajs1_emb_cell, batch_first = False).to(device) # [seq_len, batch_size, emb_dim]
    trajs2_emb_cell = pad_sequence(trajs2_emb_cell, batch_first = False).to(device) # [seq_len, batch_size, emb_dim]

    trajs1_len = torch.tensor(list(map(len, trajs1_cell)), dtype = torch.long, device = device)
    trajs2_len = torch.tensor(list(map(len, trajs2_cell)), dtype = torch.long, device = device)

    # return: two padded tensors and their lengths
    return trajs1_emb_cell, trajs1_emb_p, trajs1_len, trajs2_emb_cell, trajs2_emb_p, trajs2_len

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


# nohup python TrajCL_trajsimi_train.py --dataset xian --trajsimi_measure dtw --seed 2000 --debug &> ../result &
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

    trajcl = TrajCLTrainer()
    trajcl.cl_pretrain()
    trajcl.train_trajsimi()

