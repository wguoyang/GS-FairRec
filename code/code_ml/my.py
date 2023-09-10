# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn

import argparse
import os
import numpy as np
import math
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0]))
# print('0000')
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.metrics import f1_score
import scipy.sparse as sp
import pdb
import copy
from collections import defaultdict
import time
import data_utils
from shutil import copyfile
import pickle
# import layers#LineGCN, AvgReadout, Discriminator
import filter_layer
import pc_gender_train

##movieLens-1M
user_num=359347#user_size
item_num=292589 # item_size

user_feature=np.load('../../data/lastfm/data1t5/users_features.npy',allow_pickle=True)#(6040,30)
user_feature=user_feature.astype(np.float32)
colsum=user_feature.sum(axis=0)
colsum=colsum/user_num
trainUser,trainItem=[],[]
train_rating_dict,count=np.load('../../data/lastfm/data1t5/training_ratings_dict.npy',allow_pickle=True)#[uid,rat,iid]id从0开始
for i in train_rating_dict.keys():
    [userid, rat, itemid] = train_rating_dict[i]
    trainUser.append(userid)
    trainItem.append(itemid)
UserItemNet = csr_matrix((np.ones(len(trainUser)), (trainUser, trainItem)),
                                       shape=(user_num, item_num),dtype=np.float32)
ItemUser_mat=UserItemNet.T
rowsum=np.array(ItemUser_mat.sum(axis=1))
rowsum=np.power(rowsum,-1).flatten()
rowsum[np.isinf(rowsum)] = 0.
sum_mat=sp.diags(rowsum)
item_feature=ItemUser_mat.dot(user_feature)
item_feature=sum_mat.dot(item_feature)
colsum=colsum.reshape((1,5))
cat=np.concatenate((item_feature,colsum))
one=np.ones(shape=(292589,1),dtype=float)
E=sp.diags(one.flatten())
fone=-1*one
sprow,spcol=[],[]
for i in range(item_num):
    sprow.append(i)
    spcol.append(0)
fone=csr_matrix((fone.flatten(),(sprow,spcol)),shape=(item_num,1),dtype=np.float32)
leftm=sp.hstack((E,fone))
#leftm=np.concatenate((E,fone),axis=1)
result=leftm.dot(cat)
print(result.shape)
print(result)
np.save('../../data/lastfm/data1t5/item_feature.npy',result)
