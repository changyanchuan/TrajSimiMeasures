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

from config import Config as Config
from utilities import tool_funcs
from nn.TrjSR import TrjSR, input_processing


trjsr_epochs = 5
trjsr_batch_size = 64
trjsr_learning_rate = 0.0005
trjsr_training_bad_patience = 5


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


class TrjSRTrainer:
    def __init__(self):
        super(TrjSRTrainer, self).__init__()
        self.device = torch.device('cuda:0')        
        self.dic_datasets = self.load_trajsimi_dataset()
        
        self.checkpoint_path = '{}/{}_trajsimi_TrjSR_{}_best{}.pt'.format(Config.snapshot_dir, \
                                    Config.dataset_prefix, Config.trajsimi_measure, Config.dumpfile_uniqueid)


    def train(self):
        logging.info("training. START! @={:.3f}".format(time.time()))
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        torch.autograd.set_detect_anomaly(True)
        
        lon_range = (Config.min_lon, Config.max_lon)
        lat_range = (Config.min_lat, Config.max_lat)
        
        self.model = TrjSR(lon_range, lat_range, Config.trjsr_imgsize_x_lr, 
                        Config.trjsr_imgsize_y_lr, Config.trjsr_pixelrange_lr,
                        Config.traj_embedding_dim)
        self.model.to(self.device)
        
        cp = torch.load('{}/{}_TrjSR_{}_best.pt'.format(Config.snapshot_dir,
                                                        Config.dataset_prefix,
                                                        Config.traj_embedding_dim))
        self.model.g.load_state_dict(cp['netG'])
        self.model.d.load_state_dict(cp['netD'])
        self.model.to(self.device)
        
        self.regression = TrajSimiRegression(Config.traj_embedding_dim)
        self.regression.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.criterion.to(self.device)
        optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.model.parameters()))+
                                    list(filter(lambda p: p.requires_grad, self.regression.parameters())), \
                                    lr = trjsr_learning_rate)

        best_hr_eval = 0.0
        best_loss_train = 10000000.0
        best_epoch = 0
        bad_counter = 0
        bad_patience = trjsr_training_bad_patience
        timetoreport = [1200, 2400, 3600] # len may change later

        for i_ep in range(trjsr_epochs):
            _time_ep = time.time()
            train_losses = []
            train_gpu = []
            train_ram = []

            self.model.train()
            self.regression.train()

            for i_batch, batch in enumerate( self.trajsimi_dataset_generator_pairs_batchi( \
                                                            self.dic_datasets['trains_img'], \
                                                            self.dic_datasets['trains_simi'], \
                                                            self.dic_datasets['max_distance'])):
                _time_batch = time.time()
                optimizer.zero_grad()

                sub_trajs_img, sub_simi = batch
                # sub_trajs_img = input_processing(sub_trajs_merc, self.model.lon_range, self.model.lat_range, 
                #                                 self.model.imgsize_x_lr, self.model.imgsize_y_lr,
                #                                 self.model.pixelrange_lr).to(self.device)
                outs = self.regression(self.model(sub_trajs_img))
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
                
                # exp of training time vs. effectiveness
                if Config.trajsimi_timereport_exp and len(timetoreport) \
                        and time.time() - training_starttime >= timetoreport[0]:
                    test_metrics = self.__test(self.dic_datasets['tests_img'], \
                                                self.dic_datasets['tests_simi'], \
                                                self.dic_datasets['max_distance'])
                    logging.info("test.      ts={}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f}".format(timetoreport[0], *test_metrics))
                    timetoreport.pop(0)
                    self.model.train()
                    self.regression.train()

            # ep debug output
            logging.info("training. i_ep={}, loss={:.4f}, @={:.3f}" \
                        .format(i_ep, tool_funcs.mean(train_losses), time.time()-_time_ep))
            
            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)

            # eval
            eval_metrics = self.__test(self.dic_datasets['evals_img'], \
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

            if bad_counter == bad_patience or i_ep + 1 == trjsr_epochs:
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
        test_metrics = self.__test(self.dic_datasets['tests_img'], \
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
    def __test(self, trajs_img, datasets_simi, max_distance):
        self.model.eval()
        self.regression.eval()
        
        traj_embs = []
        datasets_simi = torch.tensor(datasets_simi, device = self.device, dtype = torch.float) / max_distance

        for i_batch, batch in enumerate(self.trajsimi_dataset_generator_batchi(trajs_img)):
            sub_trajs_img = batch
            # sub_trajs_img = input_processing(sub_trajs_merc, self.model.lon_range, self.model.lat_range, 
            #                                     self.model.imgsize_x_lr, self.model.imgsize_y_lr,
            #                                     self.model.pixelrange_lr).to(self.device)
            outs = self.regression(self.model(sub_trajs_img))
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
        

    def trajsimi_dataset_generator_batchi(self, trajs_img):
        cur_index = 0
        len_datasets = len(trajs_img)
        
        while cur_index < len_datasets:
            end_index = cur_index + trjsr_batch_size \
                                if cur_index + trjsr_batch_size < len_datasets \
                                else len_datasets
            # sub_trajs_merc = [ [tool_funcs.meters2lonlat(p[0], p[1]) for p in trajs_merc[d_idx]] for d_idx in range(cur_index, end_index)]
            sub_trajs_img = trajs_img[cur_index: end_index].to(self.device)
            yield sub_trajs_img
            cur_index = end_index


    def trajsimi_dataset_generator_pairs_batchi(self, trajs_img, datasets_simi, max_distance):
        len_datasets = len(trajs_img)
        datasets_simi = torch.tensor(datasets_simi, device = self.device, dtype = torch.float) / max_distance
        
        count_i = 0
        batch_size = len_datasets if len_datasets < trjsr_batch_size else trjsr_batch_size
        counts = math.ceil( (len_datasets / batch_size)**2 )

        while count_i < counts:
            dataset_idxs_sample = random.sample(range(len_datasets), k = batch_size)
            # dataset_idxs_sample.sort(key = lambda idx: len(trajs_merc[idx]), reverse = True) # len descending order
            # sub_trajs_merc = [ [tool_funcs.meters2lonlat(p[0], p[1]) for p in trajs_merc[d_idx]] for d_idx in dataset_idxs_sample]
            sub_trajs_img = trajs_img[dataset_idxs_sample].to(self.device)
            sub_simi = datasets_simi[dataset_idxs_sample][:,dataset_idxs_sample]

            yield sub_trajs_img, sub_simi
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
            
            trains_img = self.trajs_merc_to_coor_to_img(trains_merc)
            evals_img = self.trajs_merc_to_coor_to_img(evals_merc)
            tests_img = self.trajs_merc_to_coor_to_img(tests_merc)

        logging.info("trajsimi dataset sizes. (trains/evals/tests={}/{}/{})" \
                    .format(len(trains_merc), len(evals_merc), len(tests_merc)))

        return {'trains_img': trains_img, 'evals_img': evals_img, 'tests_img': tests_img, \
        # return {'trains_merc': trains_merc, 'evals_merc': evals_merc, 'tests_merc': tests_merc, \
                'trains_simi': train_simis, 'evals_simi': eval_simis, 'tests_simi': test_simis, \
                'max_distance': max_distance}


    def trajs_merc_to_coor_to_img(self, trajs_merc):
        lon_range = (Config.min_lon, Config.max_lon)
        lat_range = (Config.min_lat, Config.max_lat)

        trajs_coor = [ [tool_funcs.meters2lonlat(p[0], p[1]) for p in traj] for traj in trajs_merc]
        trajs_img_tensor = input_processing(trajs_coor, lon_range, lat_range, 
                                            Config.trjsr_imgsize_x_lr, Config.trjsr_imgsize_y_lr, 
                                            Config.trjsr_pixelrange_lr)
        return trajs_img_tensor

        


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


# nohup python TrjSR_trajsimi_train.py --dataset xian --trajsimi_measure dtw --seed 2000 --debug &> ../result &
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

    trjsr = TrjSRTrainer()
    trjsr.train()



