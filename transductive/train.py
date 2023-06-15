import argparse
import json
import os
import random
from pprint import pprint

import numpy as np
import torch
from torch import nn

from load_data import DataLoader
from models.red_gnn import RED_GNN_trans
from models.wo_buffer import wo_buffer
from models.RRE_GNN_raw import RRE_GNN_raw
from models.RRE_GNN import RRE_GNN
from models.w_sigmoid import w_sigmoid
from models.w_addition import w_addition
from models.w_times import w_times
from trainers import TransductiveTrainer
from utils import Dict, select_gpu

model_dict:dict = {
        "wo_buffer": wo_buffer, # Removed extra auxiliary propagation layer
        "RRE_GNN_raw": RRE_GNN_raw,
        "RED_GNN_trans": RED_GNN_trans,
        "w_addition": w_addition, # Change the gate to use an addition operation or a multiplication operation
        "w_sigmoid":w_sigmoid, # Change the attention used to a normal sigmoid rather than softmax
        "w_times":w_times, # Change the gate to use  a multiplication operation
        "RRE_GNN": RRE_GNN, # complete model
    }

def main(Trainer, my_model, my_model_name: str, results_dir='results', epoch_num=80, extend_args = None):
    my_model, my_model_name, args = parse_args(my_model, my_model_name)

    print("now pid:", os.getpid())
    loader = DataLoader(args.data_path)
    device, gpu = gpu_setting(args)
    results_dir, opts, config_list = set_global_setting(my_model_name, results_dir, extend_args, args, loader, device, gpu)

    pprint(opts.to_dict())
    save_config(opts)
    model = my_model(opts, loader)
    model.to(device)
    model = check_dist(args, model)
    print("\nnow use ", type(model))
    trainer = Trainer(opts, loader, model=model)

    after_epoch=After_epoch()
    for epoch in range(epoch_num):
        print(f"start {epoch} epoch")
        mrr, out_str, out_dict = trainer.train_epoch()
        after_epoch(results_dir, opts, model, epoch, mrr, out_str, out_dict)
    print(after_epoch.best_str)
    save_result(opts, config_list, str(type(model)), after_epoch.best_dict, after_epoch.best_str)
    print("save results finished")

def check_dist(args, model):
    """Check for Distributed Computing Conditions

    Args:
        args (_type_): Global parameters that can be used
        model (_type_): Model

    Returns:
        _type_: The processed model, if distributed, is converted to a distributed model
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1 and args.local_rank>=0:
        print('use {} gpus!'.format(num_gpus))
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank,find_unused_parameters=True)
                                                    
    return model

class After_epoch:
    """Help the model to process after training epoch
    """
    def __init__(self) -> None:
        self.best_mrr = 0
        self.best_dict = Dict()
        self.best_str = ""
    def __call__(self,results_dir, opts, model, epoch, mrr, out_str, out_dict):
    
        with open(opts.perf_file, 'a+') as f:
            f.write(out_str)
        if mrr > self.best_mrr:
            self.best_mrr = mrr
            self.best_str = out_str
            self.best_dict = out_dict
            if opts.local_rank>=0:
                save_name = f"best_{opts.local_rank}.pth"
                self.best_str = f"rank: {opts.local_rank} |{epoch} {self.best_str}"
            else:
                save_name = 'best.pth'
            if isinstance(model,nn.parallel.DistributedDataParallel):
                torch.save(model.module.state_dict(), os.path.join(results_dir, save_name))
            else:
                torch.save(model.state_dict(), os.path.join(results_dir, save_name))
            print(str(epoch) + '\t' + self.best_str)
        if opts.local_rank>=0:
            save_name = f"latest_{opts.local_rank}.pth"
        else:
            save_name = 'best.pth'
        if isinstance(model,nn.parallel.DistributedDataParallel):
            torch.save(model.module.state_dict(), os.path.join(results_dir, save_name))
        else:
            torch.save(model.state_dict(), os.path.join(results_dir, save_name))
        return self.best_dict,self.best_str

def save_config(opts):
    with open(opts.perf_file, 'a+') as f:
        pprint(opts.to_dict(), stream=f)
    with open(opts.config_file,"w",encoding="utf8")as f:
        configs = opts.to_dict()
        configs["device"] = str(configs["device"])
        json.dump(configs,f)

def save_result(opts, config_list, model_type, best_dict, best_str):
    """Save parameters during training

    Args:
        opts (_type_): Global environment parameters
        config_list (list): parameter list
        model_type (_type_): _description_
        best_dict (dict): Evaluation results of the optimal model
        best_str (str): output text of the optimal model
    """
    result_list = [model_type] + [str(value) for key, value in best_dict.items()] + config_list
    with open(opts.perf_file, 'a+') as f:
        f.write(f"Best:{best_str}\n")
        pprint(best_dict, f)
    with open("result_statistic.md", "a") as f:
        f.write("|" + "|".join(result_list) + "|\n")
    if not os.path.exists("result_statistic.json"):
        history = []
    else:
        with open("result_statistic.json", "r", encoding="utf8") as f:
            history = json.load(f)
    opts.update(best_dict)
    final_result = opts.to_dict()
    history.append(final_result)
    with open("result_statistic.json", "w", encoding="utf8") as f:
        json.dump(history, f)

def set_global_setting(my_model_name, results_dir, extend_args, args, loader, device, gpu):
    opts = Dict()
    opts.local_rank = args.local_rank
    opts.device = device
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    opts.gpu = gpu
    opts.search_range = 65 # abandoned
    opts.compose_rate = 2 # abandoned
    opts.num_workers = 12
    
    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
    # Set personalized hyperparameters
    if  dataset == 'WN18RR':
        opts.lr = 0.0003
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.n_layer = 8
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 50
        opts.n_tbatch = 50
        opts.n_extra_layer = 3
    elif dataset == 'fb15k-237':
        opts.lr = 0.0009
        opts.decay_rate = 0.9938
        opts.lamb = 0.000080
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 6
        opts.dropout = 0.0391
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 1
        opts.n_extra_layer = 3
    elif dataset == 'nell':
        opts.lr = 0.0011
        opts.decay_rate = 0.9938
        opts.lamb = 0.000089
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 6
        opts.dropout = 0.2593
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 1
        opts.n_extra_layer = 3
    else:
        raise Exception("Undefined dataset, need to set default hyperparameters for the dataset")
    # Set general hyperparameters
    if args.n_extra_layers>0:
        opts.n_extra_layer =args.n_extra_layers
    else:
        opts.n_extra_layer = 3
    if args.n_layers > 0:
        opts.n_layer = args.n_layers
    if args.n_batch>0:
        opts.n_batch=args.n_batch
    if args.n_tbatch>0:
        opts.n_tbatch=args.n_tbatch
    if extend_args is not None:
        opts.update(extend_args)
    config_list = [opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim,
                   opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act]
    config_list = [str(item) for item in config_list]

    results_dir = os.path.join(results_dir, my_model_name, dataset)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir,exist_ok=True)
    opts.perf_file = os.path.join(results_dir, dataset + '_perf.txt')
    opts.log_file = os.path.join(results_dir, 'log.txt')
    opts.config_file = os.path.join(results_dir, 'config.json')
    # Record the parameters used
    opts.results_dir = results_dir
    opts.my_model_name = my_model_name
    opts.dataset = dataset
    return results_dir,opts,config_list

def gpu_setting(args):
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        gpu = args.local_rank
        print(" use multi gpu", gpu)
    else:
        try:
            gpu = select_gpu()
        except UnicodeDecodeError:
            gpu = 0
        device_ids = []
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)
            device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
            print('gpu:', gpu)
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    return device,gpu

def parse_args(my_model, my_model_name: str):
    parser = argparse.ArgumentParser(description="Parser for SS-GNN")
    parser.add_argument('--data_path', type=str, default='data/family/')
    parser.add_argument('--seed', type=str, default=-1)
    parser.add_argument('--n_layers', type=int, default=0,
                        help="The num of basic layers")
    parser.add_argument('--n_extra_layers', type=int, default=-1,
                        help="The num of extra layers")
    parser.add_argument('--model', type=str, default="no",
                        help="The model type")
    parser.add_argument('--tag', type=str, default="no",
                        help="Additional descriptive text for the output")
    parser.add_argument('--n_batch', type=int, default=0,
                    help="train batch size")
    parser.add_argument('--n_tbatch', type=int, default=0,
                    help="test batch size")
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    args = parser.parse_args()
    # A uniform random number seed may be required for distributed training.
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)
    #random.seed(args.seed)
    
    
    if args.model in model_dict:
        my_model = model_dict[args.model]
        my_model_name = args.model

    if args.tag != "no":
        my_model_name = my_model_name + "_" + args.tag
    return my_model,my_model_name,args


if __name__ == '__main__':
    try:
        main(TransductiveTrainer, my_model=RRE_GNN, my_model_name="SES_v25")
    except:
        import sys
        import traceback
        traceback.print_exc(file=open("error_log.txt","a"))
        sys.exit(1)
