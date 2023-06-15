# RRE-GNN

## Start

A quick instruction is given for readers to reproduce the whole process.

Requirements 

- pytorch  1.8.2+cu111
- torch_scatter 2.0.8 
- scipy-1.9.2

## For transductive reasoning

We can use the following commands to train the model and evaluate the link prediction performance of RRE-GNN on the WN18RR dataset under the transductive setting.

```    
    cd transductive
    python -W ignore train.py --data_path=data/WN18RR
```

### About distributed training
```    
python -m torch.distributed.launch --nproc_per_node=2 train.py --data_path=data/WN18RR  
```    

## For inductive reasoning


We can use the following commands to train the model and evaluate the link prediction performance of RRE-GNN on the WN18RR_v1 dataset under the inductive setting.
```    
    cd inductive
    python -W ignore train.py --data_path=data/WN18RR_v1
```    

> Note: Because the size of the summarized data set is relatively small, and there are certain differences in the data distribution of the training subset and the test subset, a single training may not necessarily be able to obtain the optimal result. You can try to train the model with the same set of hyperparameters multiple times, and choose the best result based on the validated MRR value. 

> Note: The basic training code and evaluate code are copied from REDGNN's repository.