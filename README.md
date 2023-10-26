# RUN-GNN

## Start

A quick instruction is given for readers to reproduce the whole process.

Requirements

- pytorch  1.8.2+cu111
- torch_scatter 2.0.8
- scipy-1.9.2

## For transductive reasoning

We can use the following commands to train the model and evaluate the link prediction performance of RUN-GNN on the WN18RR dataset under the transductive setting.

```
    cd transductive
    python -W ignore train.py --data_path=data/WN18RR
```

### About distributed training

```
python -m torch.distributed.launch --nproc_per_node=2 train.py --data_path=data/WN18RR  
```

## For inductive reasoning

We can use the following commands to train the model and evaluate the link prediction performance of RUN-GNN on the WN18RR_v1 dataset under the inductive setting.

```
    cd inductive
    python -W ignore train.py --data_path=data/WN18RR_v1
```

> Note: Because the size of the summarized data set is relatively small, and there are certain differences in the data distribution of the training subset and the test subset, a single training may not necessarily be able to obtain the optimal result. You can try to train the model with the same set of hyperparameters multiple times, and choose the best result based on the validated MRR value.

## Citation

If you find this code useful, please consider citing the following paper.

```
@article{wu2023towards,
  title={Towards Enhancing Relational Rules for Knowledge Graph Link Prediction},
  author={Wu, Shuhan and Wan, Huaiyu and Chen, Wei and Wu, Yuting and Shen, Junfeng and Lin, Youfang},
  journal={arXiv preprint arXiv:2310.13411},
  year={2023}
}
```

## Acknowledgement

We refer to the code of [RED-GNN](https://github.com/LARS-research/RED-GNN) and [PyG](https://github.com/pyg-team/pytorch_geometric). Thanks for their contributions.

> Note: The basic training code and evaluate code are copied from REDGNN's repository.
