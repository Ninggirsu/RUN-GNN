import os
import argparse
import random

import torch
import numpy as np
from load_data import DataLoader
from base_model import IndudctiveTrainer
from utils import select_gpu
from models.RED_GNN_induc import RED_GNN_induc
from models.RRE_GNN import RRE_GNN
from pprint import pprint
from utils import Dict
parser = argparse.ArgumentParser(description="Parser for RED-GNN")
parser.add_argument('--data_path', type=str, default='data/WN18RR_v1')
parser.add_argument('--seed', type=str, default=1234)

args = parser.parse_args()

class Options(object):
    def dict(self):
        return self.__dict__
    pass

def set_seed(seed):
    SEED = seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)  
    torch.cuda.manual_seed(SEED)  
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    # if benchmark=True, deterministic will be False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]

    opts = Dict()

    try:
        gpu = select_gpu()
    except UnicodeDecodeError:
        gpu = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    print('gpu:', gpu)

    loader = DataLoader(args.data_path)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel

    if dataset == 'WN18RR_v1':
        opts.lr = 0.0095
        opts.lamb = 0.0002
        opts.decay_rate = 0.991
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.dropout = 0.516
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 100
        opts.n_extra_layer = 1
    elif dataset == 'fb237_v1':
        opts.lr = 0.00974
        opts.lamb = 0.0003
        opts.decay_rate = 0.994
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.dropout = 0.0937
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 20
        opts.n_extra_layer = 4
    elif dataset == 'nell_v1':
        opts.lr = 0.0021
        opts.lamb = 0.000189
        opts.decay_rate = 0.9937
        opts.hidden_dim = 32
        opts.attn_dim = 5
        opts.dropout = 0.3
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 10
        opts.n_extra_layer = 0
    elif dataset == 'WN18RR_v2':
        opts.lr = 0.00308
        opts.lamb = 0.0004
        opts.decay_rate = 0.994
        opts.hidden_dim = 64
        opts.attn_dim = 3
        opts.dropout = 0.200
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 20
        opts.n_extra_layer = 2
    elif dataset == 'fb237_v2':
        opts.lr = 0.00664
        opts.lamb = 0.0002
        opts.decay_rate = 0.993
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.dropout = 0.3057
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 10
        opts.n_extra_layer = 1
        
    elif dataset == 'nell_v2':
        opts.lr = 0.00872
        opts.lamb = 0.000066
        opts.decay_rate = 0.9996
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.dropout = 0.2048
        opts.act = 'relu'
        opts.n_layer = 6
        opts.n_batch = 100
        opts.n_extra_layer = 0
    elif dataset == 'WN18RR_v3':
        
        opts.lr = 0.002217
        opts.lamb = 0.000034
        opts.decay_rate = 0.991
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.dropout = 0.067
        opts.act = 'tanh'
        opts.n_layer = 7
        opts.n_batch = 20
        opts.n_extra_layer = 1
    elif dataset == 'fb237_v3':
        opts.lr = 0.00725
        opts.lamb = 0.000023
        opts.decay_rate = 0.994
        opts.hidden_dim = 64
        opts.attn_dim = 3
        opts.dropout = 0.2708
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 20
        opts.n_extra_layer = 1
    elif dataset == 'nell_v3':
        opts.lr = 0.00381
        opts.lamb = 0.0004
        opts.decay_rate = 0.995
        opts.hidden_dim = 32
        opts.attn_dim = 3
        opts.dropout = 0.089
        opts.act = 'relu'
        opts.n_layer = 7
        opts.n_batch = 10
        opts.n_extra_layer = 4
    elif dataset == 'WN18RR_v4':
        opts.lr = 0.00274
        opts.lamb = 0.000132
        opts.decay_rate = 0.991
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.2513
        opts.act = 'relu'
        opts.n_layer = 7
        opts.n_batch = 10
        opts.n_extra_layer = 3
    elif dataset == 'fb237_v4':
        opts.lr = 0.0056
        opts.lamb = 0.000018
        opts.decay_rate = 0.999
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.01904
        opts.act = 'relu'
        opts.n_layer = 6
        opts.n_extra_layer = 1
        opts.n_batch = 20
    elif dataset == 'nell_v4':
        opts.lr = 0.00639
        opts.lamb = 0.000398
        opts.decay_rate = 1
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.dropout = 0.1037
        opts.act = 'relu'
        opts.n_layer = 4
        opts.n_batch = 20
        opts.n_extra_layer = 4

    opts.act = 'relu'
    opts.n_extra_layer = 0
    
    my_model = RRE_GNN
    my_model_name = "RRE_GNN"
    results_dir = 'results'
    epoch_num = 60
    
    config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s,%s\n' % (
        opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout,
        opts.act, str(my_model))
    config_list = [opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim,
                   opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act]
    config_list = [str(item) for item in config_list]

    print(config_str)
    results_dir = os.path.join(results_dir, my_model_name, dataset)
    best_model_path = os.path.join(results_dir, 'best.pth')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    opts.save_path = results_dir
    opts.perf_file = os.path.join(results_dir, dataset + '_perf.txt')
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)
        pprint(opts.to_dict(),stream=f)
    model = my_model(opts, loader)
    pprint(opts.to_dict())
    
    print("\nnow use ", type(model))
    trainer = IndudctiveTrainer(opts, loader, model=model)

    best_mrr = 0
    best_dict = None
    best_str = ""
    for epoch in range(epoch_num):
        print(f"start {epoch} epoch")
        mrr, out_str, out_dict = trainer.train_batch()
        with open(opts.perf_file, 'a+') as f:
            f.write(out_str)
        if float(mrr)>1-1e-3:
            # Exclude the phenomenon of NaN caused by training failure, and give up the experiment
            metrics = {"default":float(best_mrr)}
            metrics.update(out_dict)
            break
        metrics = {"default":float(mrr)}
        metrics.update(out_dict)
        if mrr > best_mrr:
            best_mrr = mrr
            best_str = out_str
            best_dict = out_dict
            print(str(epoch) + '\t' + best_str)
            torch.save(model.state_dict(), best_model_path)

    print(best_str)
    metrics = {"default":float(best_mrr)}
    metrics.update(best_dict)
    result_list = [str(type(model))] + [str(value) for key,
                                        value in best_dict.items()] + config_list
    with open(opts.perf_file, 'a+') as f:
        f.write("Best:" + best_str)
    with open("result_statistic.md", "a") as f:
        f.write("|"+"|".join(result_list) + "|\n")
    print("save results finished")
