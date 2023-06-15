import torch
import numpy as np
import time

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from models.red_gnn import RED_GNN_trans
from utils import cal_ranks, cal_performance, get_logger
from torch import nn
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import RandomSampler
from load_data import DatasetWrapper,DatasetTestWrapper
from torch.utils.data import Dataset as tDataset, DataLoader as tDataLoader
from load_data import DataLoader

class TransductiveTrainer(object):
    def __init__(self, args, loader, model= None):
        if model is None:
            self.model = RED_GNN_trans(args, loader)
        else:
            self.model = model
        if torch.cuda.is_available():
            self.model.cuda()

        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_rel = loader.n_rel
        self.n_batch = args.n_batch
        self.n_tbatch = args.n_tbatch

        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test = loader.n_test
        self.n_layer = args.n_layer
        self.args = args

        self.optimizer = Adam(self.model.parameters(),
                              lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.smooth = 1e-5
        self.t_time = 0
        self.logger = get_logger(args.log_file)
        self.epoch = 0
    def train_epoch(self, ):
        epoch_loss = 0
        i = 0

        batch_size = self.n_batch
        n_batch = self.loader.n_train // batch_size + \
                  (self.loader.n_train % batch_size > 0)

        t_time = time.time()
        self.model.train()
        train_datasets = DatasetWrapper(self.loader.get_batch(np.arange(self.loader.n_train))) 
        if self.args.local_rank>=0:
            train_sampler = DistributedSampler(train_datasets)
            train_sampler.set_epoch(self.epoch)
        else:
            train_sampler = RandomSampler(train_datasets)
        train_dataloader = tDataLoader(train_datasets, sampler=train_sampler, batch_size=batch_size,
                                        num_workers=1, pin_memory=True)
        batch_num = len(train_dataloader)
        for i, triple in enumerate(train_dataloader):
            b_time = time.time()
            scores = self.model(triple[:, 0], triple[:, 1])
            if torch.cuda.is_available():
                a = torch.arange(len(scores),dtype=torch.long).cuda()
                b = torch.LongTensor(triple[:, 2]).cuda()
                indexes = [a, b]
                pos_scores = scores[indexes]
            else:
                pos_scores = scores[[torch.arange(len(scores)), torch.LongTensor(triple[:, 2])]]

            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(- pos_scores + max_n +
                             torch.log(torch.sum(torch.exp(scores - max_n), 1)))
            loss.backward()
            self.optimizer.step()
            self.model.zero_grad()

            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            if (i+1)%10==0:
                epoch_loss += loss.item()
                avg_time_cost = (time.time()-t_time)/(i+1)
                remain_batch = batch_num-i
                remain_time = avg_time_cost*remain_batch
                self.logger.info(f"{i}/{batch_num} batch finished, total use {time.time()-b_time:.4f} s. etc: {remain_time:.2f}")
        self.scheduler.step()
        self.t_time += time.time() - t_time

        valid_mrr, out_str, out_dict = self.evaluate()
        self.loader.shuffle_train(epoch=self.epoch)
        self.epoch = self.epoch+1
        return valid_mrr, out_str, out_dict

    def evaluate(self, ):
        batch_size = self.n_tbatch

        n_data = self.n_valid
        ranking = []
        self.model.eval()
        i_time = time.time()
        valid_datasets = DatasetTestWrapper(self.loader.get_batch(np.arange(n_data),data="valid")) 
        if self.args.local_rank>=0:
            valid_sampler = DistributedSampler(valid_datasets)
        else:
            valid_sampler = RandomSampler(valid_datasets)
        valid_dataloader = tDataLoader(valid_datasets, sampler=valid_sampler, batch_size=batch_size,
                                        num_workers=1, pin_memory=True)
        batch_num = len(valid_dataloader)
        self.logger.info(f"start valid process {self.args.local_rank}")
        for i, triple in enumerate(valid_dataloader):
            b_time = time.time()
            subs, rels, objs = triple
            scores = self.model(subs, rels, mode='valid').data.cpu().numpy()
            filters = []
            if isinstance(subs,torch.Tensor):
                subs = subs.numpy()
                rels = rels.numpy()
                objs = objs.numpy()
            for j in range(len(subs)):
                
                filt = self.loader.filters[(subs[j], rels[j])]
                filt_1hot = np.zeros((self.n_ent,))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)

            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
            if (i+1)%100==0:
                avg_time_cost = (time.time()-i_time)/(i+1)
                remain_batch = batch_num-i
                remain_time = avg_time_cost*remain_batch
                self.logger.info(f"valid {i}/{batch_num} batch finished, total use {time.time()-b_time:.4f} s. etc: {remain_time:.2f}")
        ranking = np.array(ranking)
        v_mrr, v_h1, v_h3, v_h10, v_h100 = cal_performance(ranking)
        # syndata 
        if self.args.local_rank>=0:
            self.logger.info(f"start valid process syn at {self.args.local_rank}")
            perf = torch.Tensor((v_mrr, v_h1, v_h3, v_h10, v_h100 )).cuda()
            dist.all_reduce(perf, op=dist.ReduceOp.SUM)
            perf = (perf/dist.get_world_size()).cpu().tolist()
            v_mrr, v_h1, v_h3, v_h10, v_h100 = perf
            self.logger.info(f"finish valid process syn at {self.args.local_rank}")
            
        n_data = self.n_test
        ranking = []
        self.model.eval()
        test_datasets = DatasetTestWrapper(self.loader.get_batch(np.arange(n_data),data="test")) 
        if self.args.local_rank>=0:
            test_sampler = DistributedSampler(test_datasets)
        else:
            test_sampler = RandomSampler(test_datasets)
        test_dataloader = tDataLoader(test_datasets, sampler=test_sampler, batch_size=batch_size,
                                        num_workers=1, pin_memory=True)
        batch_num = len(test_dataloader)
        self.logger.info(f"start test process {self.args.local_rank}")
        for i, triple in enumerate(test_dataloader):
            b_time = time.time()
            subs, rels, objs = triple
            scores = self.model(subs, rels, mode='test').data.cpu().numpy()
            filters = []
            if isinstance(subs,torch.Tensor):
                subs = subs.numpy()
                rels = rels.numpy()
                objs = objs.numpy()
            for i in range(len(subs)):
                filt = self.loader.filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent,))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)

            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
            if (i+1)%100==0:
                avg_time_cost = (time.time()-i_time)/(i+1)
                remain_batch = batch_num-i
                remain_time = avg_time_cost*remain_batch
                self.logger.info(f"test {i}/{batch_num} batch finished, total use {time.time()-b_time:.4f} s. etc: {remain_time:.2f}")
        ranking = np.array(ranking)
        t_mrr, t_h1, t_h3, t_h10, t_h100 = cal_performance(ranking)
        # syndata 
        if self.args.local_rank>=0:
            self.logger.info(f"start valid process syn at {self.args.local_rank}")
            perf = torch.Tensor((t_mrr, t_h1, t_h3, t_h10, t_h100 )).cuda()
            dist.all_reduce(perf, op=dist.ReduceOp.SUM)
            perf = (perf/dist.get_world_size()).cpu().tolist()
            t_mrr, t_h1, t_h3, t_h10, t_h100 = perf
            self.logger.info(f"finish valid process syn at {self.args.local_rank}")
        i_time = time.time() - i_time

        out_str = '[VALID] MRR:%.4f H@1:%.4f H@3:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@3:%.4f H@10:%.4f \t[TIME] train:%.4f inference:%.4f\n' % (
            v_mrr, v_h1,v_h3, v_h10, t_mrr, t_h1, t_h3, t_h10, self.t_time, i_time)
        out_str = f"[epoch {self.epoch:3d}]{out_str}"
        out_dict = {"valid MRR": v_mrr, "valid Hit@1": v_h1, "valid Hit@3": v_h3, "valid Hit@10": v_h10,
                    "valid Hit@100": v_h100,
                    "test MRR": t_mrr, "test Hit@1": t_h1, "test Hit@3": t_h3, "test Hit@10": t_h10,
                    "test Hit@100": t_h100,
                    "training time": self.t_time, "inference time": i_time}
        return v_mrr, out_str, out_dict
    