import torch
import numpy as np
import time

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from utils import cal_ranks, cal_performance,cal_ranks_origin

class IndudctiveTrainer(object):
    def __init__(self, args, loader, model, use_gpu=True):
        self.use_gpu = use_gpu if torch.cuda.is_available() else False
        self.model = model
        if use_gpu:
            self.model.cuda()
        self.device = next(self.model.parameters()).device
        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_ent_ind = loader.n_ent_ind
        self.n_batch = args.n_batch

        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test  = loader.n_test
        self.n_layer = args.n_layer

        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.t_time = 0

    def train_batch(self,):
        epoch_loss = 0
        i = 0

        batch_size = self.n_batch
        n_batch = self.n_train // batch_size + (self.n_train % batch_size > 0)

        t_time = time.time()
        self.model.train()
        for i in range(n_batch):
            start = i*batch_size
            end = min(self.n_train, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx)

            self.model.zero_grad()
            scores = self.model(triple[:,0], triple[:,1])

            pos_scores = scores[[torch.arange(len(scores)).to(self.device),torch.LongTensor(triple[:,2]).to(self.device)]]
            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1)))
            loss.backward()
            self.optimizer.step()

            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()
        self.scheduler.step()
        self.t_time += time.time() - t_time
        self.loader.shuffle_train()
        valid_mrr, out_str,out_dict = self.evaluate()
        return valid_mrr, out_str, out_dict

    def evaluate(self, ):
        batch_size = self.n_batch

        n_data = self.n_valid
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        i_time = time.time()
        for i in range(n_batch):
            start = i*batch_size
            end = min(n_data, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='valid')
            scores = self.model(subs, rels).data.cpu().numpy()
            filters = []
            for i in range(len(subs)):
                filt = self.loader.val_filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent, ))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)
             
            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        v_mrr, v_h1,v_h3, v_h10 = cal_performance(ranking)

        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        for i in range(n_batch):
            start = i*batch_size
            end = min(n_data, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='test')
            scores = self.model(subs, rels, 'inductive').data.cpu().numpy()
            filters = []
            for i in range(len(subs)):
                filt = self.loader.tst_filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent_ind, ))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)
             
            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        t_mrr, t_h1,t_h3, t_h10 = cal_performance(ranking)
        i_time = time.time() - i_time

        out_str = '[VALID] MRR:%.4f H@1:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@10:%.4f \t[TIME] train:%.4f inference:%.4f\n' % (
            v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10, self.t_time, i_time)
        out_dict = {"valid MRR": v_mrr, "valid Hit@1": v_h1, "valid Hit@3": v_h3, "valid Hit@10": v_h10,
                    "test MRR": t_mrr, "test Hit@1": t_h1, "test Hit@3": t_h3, "test Hit@10": t_h10,
                    "training time": self.t_time, "inference time": i_time}
        return v_mrr, out_str, out_dict

    def evaluate_old(self, ):
        batch_size = self.n_batch

        n_data = self.n_valid
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        i_time = time.time()
        for i in range(n_batch):
            start = i*batch_size
            end = min(n_data, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='valid')
            scores = self.model(subs, rels).data.cpu().numpy()
            filters = []
            for i in range(len(subs)):
                filt = self.loader.val_filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent, ))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)
             
            filters = np.array(filters)
            ranks = cal_ranks_origin(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        v_mrr, v_h1,v_h3, v_h10 = cal_performance(ranking)

        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        for i in range(n_batch):
            start = i*batch_size
            end = min(n_data, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='test')
            scores = self.model(subs, rels, 'inductive').data.cpu().numpy()
            filters = []
            for i in range(len(subs)):
                filt = self.loader.tst_filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent_ind, ))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)
             
            filters = np.array(filters)
            ranks = cal_ranks_origin(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        t_mrr, t_h1,t_h3, t_h10 = cal_performance(ranking)
        i_time = time.time() - i_time

        out_str = '[VALID] MRR:%.4f H@1:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@10:%.4f \t[TIME] train:%.4f inference:%.4f\n' % (
            v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10, self.t_time, i_time)
        out_dict = {"valid MRR": v_mrr, "valid Hit@1": v_h1, "valid Hit@3": v_h3, "valid Hit@10": v_h10,
                    "test MRR": t_mrr, "test Hit@1": t_h1, "test Hit@3": t_h3, "test Hit@10": t_h10,
                    "training time": self.t_time, "inference time": i_time}
        return v_mrr, out_str, out_dict

