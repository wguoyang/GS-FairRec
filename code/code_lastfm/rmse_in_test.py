import random

import torch
# print(torch.__version__)
import torch.nn as nn

import argparse
import os
import numpy as np

import math
import sys

#os.environ["CUDA_VISIBLE_DEVICES"] ='3'
# print('0000')
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd

from sklearn import metrics
from sklearn.metrics import f1_score
import pdb
import copy
from collections import defaultdict
import time
import data_utils
from shutil import copyfile
import pickle

# import layers#LineGCN, AvgReadout, Discriminator
import filter_layer
import pc_age_train
# user_num=359347#user_size
# item_num=292589#item_size
# user=np.zeros(shape=(1,user_num),dtype=int)
# item=np.zeros(shape=(1,item_num),dtype=int)
# user_sample,item_sample=[],[]
#
# for i in range(5000):
#     u=random.randint(0,359347)
#     if user[0,u]==0:
#         user[0,u]=1
#         user_sample.append(u)
#     v = random.randint(0, 292589)
#     if item[0, v] == 0:
#         item[0, v] = 1
#         item_sample.append(v)
users_emb_gcn = np.load('./gcnModel/user_embedding_222.npy',allow_pickle=True)
items_emb_gcn = np.load('./gcnModel/item_embedding_222.npy',allow_pickle=True)
FairMetric=data_utils.FairAndPrivacy()
precision_Att=FairMetric.get_Att_precision(users_emb_gcn,items_emb_gcn,'gender')
precision_Att=round(precision_Att,4)
#fairscore=round(fairscore/test_dict_count,4)
print(precision_Att)
# np.save('../../data/lastfm/data1t5/RecFair_user_sample.npy',user_sample)
# np.save('../../data/lastfm/data1t5/RecFair_item_sample.npy',item_sample)
import pc_gender_train
# user_num=359347#user_size
# item_num=292589#item_size
# factor_num=64
# shiyanmiaoshu='LightGCN+NF_0.20的性别，年龄AUCF1，rmse'
# f = open('lightGCN__AUC_F1_RMSE_result.txt', 'a')
# f.write(shiyanmiaoshu+'\n')
# f.close()
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
# users_emb_gcn = np.load('gcnModel/user_emb_epoch50.npy', allow_pickle=True)
# print(users_emb_gcn.shape)
# #users_emb_gcn=torch.cuda.FloatTensor(users_emb_gcn)
# #users_emb_gcn是一个矩阵6040行64列，也就是embedding向量是64维的
# items_emb_gcn = np.load('gcnModel/item_emb_epoch50.npy', allow_pickle=True)
# print(items_emb_gcn.shape)
#users_emb_lightgcn = np.load('gcnModel/user_embedding_222.npy', allow_pickle=True)
# user_adj=np.load('Node_adj_score/USERCF_user_adj.npy',allow_pickle=True).item()
# item_adj=np.load('Node_adj_score/USERCF_item_adj.npy',allow_pickle=True).item()

#users_emb_gcn=torch.cuda.FloatTensor(users_emb_gcn)
#users_emb_gcn是一个矩阵6040行64列，也就是embedding向量是64维的
#items_emb_lightgcn = np.load('gcnModel/item_embedding_222.npy', allow_pickle=True)
# users_emb_gcn = np.load('./gcnModel/L_eui_user_embedding_NF_0.20_26.npy',allow_pickle=True)
# items_emb_gcn = np.load('./gcnModel/L_eui_item_embedding_NF_0.20_26.npy',allow_pickle=True)
# usernotintrainset,item_not_in_trainset=np.load('gcnModel/node_not_in_trainset.npy',allow_pickle=True)
# for itemid in item_not_in_trainset:
#     items_emb_lightgcn[itemid]=users_emb_gcn[itemid]
# for userid in usernotintrainset:
#     users_emb_lightgcn[userid]=users_emb_gcn[userid]
# str_f1f1_res=''
# auc_one,auc_res= pc_gender_train.clf_gender_all_pre(0,0,users_emb_gcn,factor_num)
# str_f1f1_res+='\t gender auc:'+str(round(auc_one,4))+'\t'
# print(str_f1f1_res)
# f1p_one, f1r_one, f1res_p, f1res_r = pc_age_train.clf_age_all_pre(0, 0, users_emb_gcn, factor_num)
# f1micro_f1 = (2 * f1p_one * f1r_one) / (f1p_one + f1r_one)
# str_f1f1_res += 'age f1:' + str(round(f1micro_f1, 4)) + '\t'
# print(str_f1f1_res)
# user_not_in_trainset,item_not_in_trainset=[],[]
# #
# train_ratings_dict,train_dict_count = np.load('../../data/lastfm/data1t5/training_ratings_dict.npy',allow_pickle=True)
# user=np.zeros(shape=(1,user_num),dtype=int)
# item=np.zeros(shape=(1,item_num),dtype=int)
# for i in train_ratings_dict.keys():
#     [userid, rat, itemid] = train_ratings_dict[i]
#     user[0,userid]=1
#     item[0,itemid]=1
# for userid in range(user_num):
#     if user[0,userid]==0:
#         user_not_in_trainset.append(userid)
# for itemid in range(item_num):
#     if item[0,itemid]==0:
#         item_not_in_trainset.append(itemid)
# node_not_in_trainset=[user_not_in_trainset,item_not_in_trainset]
# np.save('gcnModel/node_not_in_trainset.npy',node_not_in_trainset)
testing_ratings_dict,test_dict_count = np.load('../../data/lastfm/data1t5/testing_ratings_dict.npy',allow_pickle=True)
# new_test_set={}
# count=0
# for i in testing_ratings_dict.keys():
#     [userid, rat, itemid] = testing_ratings_dict[i]
#     if item[0,itemid]==1:
#         new_test_set[count]=[userid, rat, itemid]
#         count+=1

#np.save('../../data/lastfm/data1t5/new_testing_ratings_dict.npy',new_test_set)
#

user_e=users_emb_gcn
item_e=items_emb_gcn
pre_all = []#保存预测打分
label_all = []#保存实际打分
for pair_i in testing_ratings_dict:
    u_id, r_v, i_id = testing_ratings_dict[pair_i]#[用户id，评分，电影id]
    pre_get = np.sum(user_e[u_id]*item_e[i_id])
    #fairscore = fairscore + pre_get * Fairness.get_Fairness_Score(userid=u_id, itemid=i_id, Attribute='gender')
    pre_all.append(pre_get)
    label_all.append(r_v)
r_test=rmse(np.array(pre_all),np.array(label_all))#计算预测和实际打分的rmse
res_test=round(np.mean(r_test),4)#保留四位小数
print(res_test)
# str_f1f1_res +='RMSE:' + str(res_test) + '\n'
# f = open('lightGCN__AUC_F1_RMSE_result.txt', 'a')
# f.write(str_f1f1_res+'\n')
# f.close()


