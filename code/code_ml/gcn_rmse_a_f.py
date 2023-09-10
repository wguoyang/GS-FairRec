import torch
# print(torch.__version__)
import torch.nn as nn

import argparse
import os
import numpy as np
import testprocedure
import math
import sys

os.environ["CUDA_VISIBLE_DEVICES"] ='1'
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
import utils
# import layers#LineGCN, AvgReadout, Discriminator
import filter_layer
import pc_age_train
import pc_occ_train
import pc_gender_train
user_num=6040#user_size
item_num=3952#item_size
factor_num=64
top_k=20
# a=np.ones(shape=(2,10),dtype=float)
# b=3
# a[0]=b*a[0]
# a[0]=a[0]/2
# print(a)

Fairness=utils.FairnessScore()
FairMetric=utils.FairAndPrivacy()


def rankrep( user_e, item_e):
    user_e_new = np.zeros(shape=(user_num, 64), dtype=float)
    user_e_new = torch.cuda.FloatTensor(user_e_new)
    user_e = torch.cuda.FloatTensor(user_e)
    item_e = torch.cuda.FloatTensor(item_e)
    for i in range(user_num):
        predict = np.zeros(shape=(1, item_num), dtype=float)
        user_new = np.zeros(shape=(1, 64), dtype=float)
        predict = torch.cuda.FloatTensor(predict)
        user_new = torch.cuda.FloatTensor(user_new)
        for j in range(item_num):
            predict[0][j] = torch.sum(user_e[i] * item_e[j])
        maxtoppre = np.zeros(shape=(1, top_k), dtype=float)
        maxtopi = np.zeros(shape=(1, top_k), dtype=int)
        tag = np.zeros(shape=(1, item_num), dtype=int)
        maxtoppre = torch.cuda.FloatTensor(maxtoppre)
        for top in range(top_k):
            maxp = 0
            maxi = 0
            for j in range(item_num):
                if predict[0][j] > maxp:
                    if tag[0][j] == 0:
                        maxp = predict[0][j]
                        maxi = j
                        tag[0][j] = 1
            maxtoppre[0][top] = maxp
            maxtopi[0][top] = maxi
        sump = torch.sum(maxtoppre[0])
        for top in range(top_k):
            user_new = user_new + maxtoppre[0][top] * item_e[maxtopi[0][top]]
        user_new = user_new / sump
        # print(maxtoppre)
        # print(maxtopi)
        # print(user_new)
        user_e_new[i] = user_new
    return user_e_new

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
dataset_base_path='../../data/ml1m'
users_emb_gcn = np.load('gcnModel/LGN_MLP_user_embedding_34.npy', allow_pickle=True)
print(users_emb_gcn.shape)
#users_emb_gcn=torch.cuda.FloatTensor(users_emb_gcn)
#users_emb_gcn是一个矩阵6040行64列，也就是embedding向量是64维的
items_emb_gcn = np.load('gcnModel/LGN_MLP_item_embedding_34.npy', allow_pickle=True)
#items_emb_gcn = np.load('gcnModel/item_emb_epoch79.npy', allow_pickle=True)
print(items_emb_gcn.shape)
# fair=FairMetric.get_Att_precision(users_emb_gcn,items_emb_gcn,'occ')
# print(fair)
#items_emb_gcn=torch.cuda.FloatTensor(items_emb_gcn)
#items_emb_gcn是一个矩阵3952行64列

###precision...
# tester=testprocedure.TestPcd()
# tester.Test(users_emb_gcn,items_emb_gcn)
###

#rmse
"""


"""
testing_ratings_dict,test_dict_count = np.load(dataset_base_path+'/data1t5/testing_ratings_dict.npy',allow_pickle=True)

pre_all = []#保存预测打分
label_all = []#保存实际打分
user_e=users_emb_gcn
item_e=items_emb_gcn
fairscore=0
for pair_i in testing_ratings_dict:
    u_id, r_v, i_id = testing_ratings_dict[pair_i]#[用户id，评分，电影id]
    r_v+=1
    pre_get = np.sum(user_e[u_id]*item_e[i_id])
    #fairscore = fairscore + pre_get * Fairness.get_Fairness_Score(userid=u_id, itemid=i_id, Attribute='gender')
    pre_all.append(pre_get)
    label_all.append(r_v)
r_test=rmse(np.array(pre_all),np.array(label_all))#计算预测和实际打分的rmse
res_test=round(np.mean(r_test),4)#保留四位小数
print(res_test)#0.8564
#print(fairscore/test_dict_count)
#age,f1
"""


str_print_evl+=str_f1f1_res
f1_lsit=(2*np.array(f1res_p)*np.array(f1res_r))/(np.array(f1res_p)+np.array(f1res_r))
for i_f1 in f1_lsit:
    str_print_evl+=str(round(i_f1,4))+' '
"""



# f1p_one,f1r_one,f1res_p,f1res_r= pc_age_train.clf_age_all_pre('gcn_age_f1',-1,users_emb_gcn,64)
# f1pone=np.mean(f1res_p)
# f1rone=np.mean(f1res_r)
# f1micro_f1=(2*f1pone*f1rone)/(f1pone+f1rone)
# str_f1f1_res='age f1:'+str(round(f1micro_f1,4))+'\t'
# print(str_f1f1_res)#0.4306,0.4214,0.4493,

#gender,auc
"""

"""
#training_ratings_dict,train_dict_count = np.load(dataset_base_path+'/data1t5/training_ratings_dict.npy',allow_pickle=True)
# training_u_i_set,training_i_u_set = np.load(dataset_base_path+'/data1t5/training_adj_set.npy',allow_pickle=True)
# g_adj= data_utils.generate_adj(training_u_i_set,training_i_u_set,user_num,item_num)
# pos_adj=g_adj.generate_pos()
# pos_adj=data_utils.adj_mairx(training_ratings_dict,ifRW=False)
# gcn_items_embedding0 = torch.cuda.FloatTensor(items_emb_gcn)
# users_f1_local = torch.sparse.mm(pos_adj, gcn_items_embedding0)
# first_order=users_f1_local.cpu().numpy()
# user_e=np.load('ego_graph/user_ego_graph1.npy',allow_pickle=True)
#user_e[4168]=users_emb_gcn[4168]

# f1p_one,f1r_one,f1res_p,f1res_r= pc_age_train.clf_age_all_pre('gcn_age_f1',-1,first_order,64)
# f1pone=np.mean(f1res_p)
# f1rone=np.mean(f1res_r)
# f1micro_f1=(2*f1pone*f1rone)/(f1pone+f1rone)
# print(f1micro_f1)
#
# f1p_one,f1r_one,f1res_p,f1res_r= pc_occ_train.clf_occ_all_pre('gcn_occ_f1',-1,first_order,64)
# f1pone=np.mean(f1res_p)
# f1rone=np.mean(f1res_r)
# f1micro_f1=(2*f1pone*f1rone)/(f1pone+f1rone)
# print(f1micro_f1)
gender,age,occ=[],[],[]
for i in range(20):
    auc_one,auc_res=pc_gender_train.clf_gender_all_pre('gcn_age_auc',-1,users_emb_gcn,64)
    if np.mean(auc_res)>0.65:
        gender.append(np.mean(auc_res))
    str_f1f1_res='\t gender auc:'+str(round(np.mean(auc_res),4))+'\t'
    print(str_f1f1_res)#0.8338,0.8248,0.7863,0.8009,0.8435,0.8167,0.8211
    f1p_one, f1r_one, f1res_p, f1res_r = pc_age_train.clf_age_all_pre('gcn_age_f1', -1, users_emb_gcn, 64)
    f1pone = np.mean(f1res_p)
    f1rone = np.mean(f1res_r)
    f1micro_f1 = (2 * f1pone * f1rone) / (f1pone + f1rone)
    if f1micro_f1>0.4:
        age.append(f1micro_f1)
    print(f1micro_f1)
    f1p_one, f1r_one, f1res_p, f1res_r = pc_occ_train.clf_occ_all_pre('gcn_occ_f1', -1, users_emb_gcn, 64)
    f1pone = np.mean(f1res_p)
    f1rone = np.mean(f1res_r)
    f1micro_f1 = (2 * f1pone * f1rone) / (f1pone + f1rone)
    if f1micro_f1>0.14:
        occ.append(f1micro_f1)
    print(f1micro_f1)
    # #0.7914,0.7879,0.7793,0.6274,0.8059,0.8261,0.6742,0.8048,0.5784,0.7739
    # #0.8176,0.7569,0.7345,0.7429,0.8043,0.7964
    # f1p_one,f1r_one,f1res_p,f1res_r= pc_age_train.clf_age_all_pre('gcn_age_f1',-1,users_emb_gcn,64)
    # f1pone=np.mean(f1res_p)
    # f1rone=np.mean(f1res_r)
    # f1micro_f1=(2*f1pone*f1rone)/(f1pone+f1rone)
    # sum = sum + f1micro_f1
    # str_f1f1_res='age f1:'+str(round(f1micro_f1,4))+'\t'
    # print(str_f1f1_res)#0.4306,0.4214,0.4493,
    # f1p_one,f1r_one,f1res_p,f1res_r= pc_occ_train.clf_occ_all_pre('gcn_age_f1',-1,first_order,64)
    # f1pone=np.mean(f1res_p)
    # f1rone=np.mean(f1res_r)
    # f1micro_f1=(2*f1pone*f1rone)/(f1pone+f1rone)
    # sum=sum+f1micro_f1
    # str_f1f1_res='occ f1:'+str(round(f1micro_f1,4))+'\t'
    # print(str_f1f1_res)#0.1593,0.176,0.177,0.1767,0.178
print()
print(np.mean(gender))
print(np.mean(age))
print(np.mean(occ))

#occ,f1
"""

"""
# if __name__ == '__main__':
# f1p_one,f1r_one,f1res_p,f1res_r= pc_occ_train.clf_occ_all_pre('gcn_age_f1',-1,users_emb_gcn,64)
# f1pone=np.mean(f1res_p)
# f1rone=np.mean(f1res_r)
# f1micro_f1=(2*f1pone*f1rone)/(f1pone+f1rone)
# str_f1f1_res='occ f1:'+str(round(f1micro_f1,4))+'\t'
# print(str_f1f1_res)#0.1593,0.176,0.177,0.1767,0.178










