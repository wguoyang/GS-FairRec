import utils
import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as F
from time import time
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
def BPR_train_original(dataset, recommend_model, loss_class):
    Recmodel = recommend_model
    Recmodel.train()
    bpr= loss_class
    S, sam_time = utils.UniformSample_original(dataset)
    print(f"BPR[sample time][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to('cuda')
    posItems = posItems.to('cuda')
    negItems = negItems.to('cuda')
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // 2048 + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=2048)):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    return f"[BPR[aver loss{aver_loss:.3e}]"