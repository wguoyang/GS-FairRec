import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import cgs
from scipy.sparse.linalg import spsolve
import multiprocessing
import os
os.environ["MKL_INTERFACE_LAYER"] = "ILP64"
from sparse_dot_mkl import dot_product_mkl

lastfm=True
isuser=False
#lastfm
if lastfm==True:
    if isuser:
        user_num=359347#user_size
        item_num=292589#item_size
    else:
        user_num=292589
        item_num=359347
    train_rating_dict, train_count = np.load('../../data/lastfm/data1t5/training_ratings_dict.npy',
                                             allow_pickle=True)
else:
    if isuser:
        user_num = 6040  # user_size
        item_num = 3952  # item_size
    else:
        user_num=3952
        item_num=6040
    train_rating_dict, train_count = np.load('../../data/ml1m/data1t5/training_ratings_dict.npy',
                                                         allow_pickle=True)
#ml1m



#############################################################user_cf#######################################################
#user_adj_dict
trainUser,trainItem,trainrating,trainrating_pow2=[],[],[],[]
user_adj_dict={}
for uid in range(user_num):
    user_adj_dict[uid]={}
for i in train_rating_dict.keys():
    if isuser:
        [userid, rat, itemid] = train_rating_dict[i]
    else:
        [itemid, rat, userid] = train_rating_dict[i]
    # ml1m需要rat+1
    if lastfm==False:
        rat=rat+1
    trainUser.append(userid)
    trainItem.append(itemid)
    trainrating.append(rat)
    user_adj_dict[userid][itemid]=0.
    trainrating_pow2.append(rat*rat)

print(max(trainrating))
traindataSize = train_count
trainUser = np.array(trainUser)
trainItem = np.array(trainItem)
trainrating = np.array(trainrating)
trainrating_pow2=np.array(trainrating_pow2)
UserItemNet = csr_matrix((trainrating, (trainUser, trainItem)),
                                      shape=(user_num, item_num),dtype=np.float32)
#UserItemNetcsc=UserItemNet.tocsc()
UserItemNet_pow2 = csr_matrix((trainrating_pow2, (trainUser, trainItem)),
                                      shape=(user_num, item_num),dtype=np.float32)
rowsum = np.array(UserItemNet_pow2.sum(axis=1))#按行求和(usernum,1)
d_inv = np.power(rowsum, -0.5).flatten()#开根号分之一(usernum,)
d_inv[np.isinf(d_inv)] = 0.
d_inv=d_inv.reshape((user_num,1))
#u0=UserItemNet[0,:]#一个稀疏向量
uid=0
batch=100
batchnum=0
shengyu=user_num%batch
print(shengyu)
while True:
    t0=time.time()
    # user=UserItemNet[uid,:]
    # user=user.todense().T#(itemnum,1)
    # cos=csr_matrix.dot(UserItemNet,user)#(usernum,1)
    # cos[uid,0]=0
    # ud_inv=d_inv*d_inv[uid]
    # ud_inv=ud_inv.reshape((user_num,1))
    # cosin=np.multiply(ud_inv,cos)#(usernum,1)
    # result= csr_matrix.dot(UserItemNet.T, cosin)
    if uid+batch>=user_num:
        batch=shengyu#ml1m是40，lastfm是47
    user=UserItemNet[uid:uid+batch,:]
    user=user.todense().T#(itemnum,1)
    cos=csr_matrix.dot(UserItemNet,user)#(usernum,1),(usernum,100)
    k=0
    for i in range(uid,uid+batch):
        #print(i)
        cos[i,k]=0
        k+=1
    user_batch=d_inv[uid:uid+batch,0]
    user_batch=user_batch.reshape((batch,1))
    ud_inv = np.dot(d_inv, user_batch.T)
    cosin = np.multiply(ud_inv, cos)  # (usernum,100)
    colsum=np.array(cosin.sum(axis=0))#(1,100)
    result = csr_matrix.dot(UserItemNet.T, cosin)#(itemnum,100)
    for index in range(uid,uid+batch):
        for iid in user_adj_dict[index]:
            user_adj_dict[index][iid]=result[iid,index-batchnum*100]/colsum[0,index-batchnum*100]

    if batch==shengyu:#ml1m是40，lastfm是47
        break
    t1=time.time()
    print(t1-t0)
    uid=uid+batch
    print(uid)
    batchnum+=1
np.save('../code_lastfm/Node_adj_score/USERCF_item_adj.npy',user_adj_dict)
# print(user_adj_dict)







##############################################################Number_of_paths_based#####################################################
# #user_adj_dict
# trainUser,trainItem,trainrating,trainrating_pow2=[],[],[],[]
# user_adj_dict={}
# for uid in range(user_num):
#     user_adj_dict[uid]={}
# for i in train_rating_dict.keys():
#     if isuser:
#         [userid, rat, itemid] = train_rating_dict[i]
#     else:
#         [itemid, rat, userid] = train_rating_dict[i]
#     # ml1m需要rat+1
#     if lastfm==False:
#         rat=rat+1
#     trainUser.append(userid)
#     trainItem.append(itemid)
#     trainrating.append(rat)
#     user_adj_dict[userid][itemid]=0.
#
# print(max(trainrating))
# traindataSize = train_count
# trainUser = np.array(trainUser)
# trainItem = np.array(trainItem)
# trainrating = np.array(trainrating)
# trainrating_pow2=np.array(trainrating_pow2)
# UserItemNet = csr_matrix((np.ones(len(trainUser)), (trainUser, trainItem)),
#                                       shape=(user_num, item_num),dtype=np.float32)
# UserItemNetcsc=UserItemNet.tocsc()
#
# #u0=UserItemNet[0,:]#一个稀疏向量
#
# for uid in range(user_num):
#     t0=time.time()
#     user=np.array(UserItemNet[uid,:].todense()).flatten()
#     user_diag=sp.diags(user)#(itemnum,itemnum)
#     user_U_I = UserItemNetcsc.dot(user_diag)
#     rowsum=np.array(user_U_I.sum(axis=1))
#     rowsum=rowsum-np.ones(rowsum.shape)
#     rowsum[uid,0]=0
#     rowsum=rowsum.flatten()
#     num_diag = sp.diags(rowsum, format='csc')
#     user_U_I = num_diag.dot(user_U_I)
#     result=np.array(user_U_I.sum(axis=0))
#     for iid in user_adj_dict[uid]:
#         user_adj_dict[uid][iid]=result[0,iid]
#     t1=time.time()
#     print(t1-t0)
#
#
# np.save('../code_ml/RW_adj_score/NoP_FGitem_adj.npy',user_adj_dict)
# print(user_adj_dict)




#######################################################itemcf_kmeans########################################################
#user_adj_dict
#
# trainUser,trainItem,trainrating,trainrating_pow2=[],[],[],[]
# user_adj_dict={}
# user_neigh={}
# for uid in range(user_num):
#     user_adj_dict[uid]={}
#     user_neigh[uid]=[]
# for i in train_rating_dict.keys():
#     if isuser:
#         [userid, rat, itemid] = train_rating_dict[i]
#     else:
#         [itemid, rat, userid] = train_rating_dict[i]
#     # ml1m需要rat+1
#     if lastfm==False:
#         rat=rat+1
#     trainUser.append(userid)
#     trainItem.append(itemid)
#     trainrating.append(rat)
#     user_adj_dict[userid][itemid]=0.
#     user_neigh[userid].append(itemid)
#
# print(max(trainrating))
# traindataSize = train_count
# trainUser = np.array(trainUser)
# trainItem = np.array(trainItem)
# trainrating = np.array(trainrating)
# UserItemNet = csr_matrix((trainrating, (trainUser, trainItem)),
#                                       shape=(user_num, item_num),dtype=np.float32)
# UserItemNetcsc=UserItemNet.tocsc()
# model=KMeans(n_clusters=1,max_iter=100)
# #u0=UserItemNet[0,:]#一个稀疏向量
#
# for uid in range(user_num):
#     t0=time.time()
#     user=user_neigh[uid]
#     num=len(user)
#     item_feature=UserItemNetcsc[:,user]
#     item_feature=item_feature.todense().T
#     tt0=time.time()
#     model.fit(item_feature)
#     tt1=time.time()
#     print(tt1-tt0)
#     center=model.cluster_centers_
#     item_feature=np.concatenate((item_feature,center))
#     one=np.ones(shape=(num,1),dtype=float)
#     E=np.diag(one.flatten())
#     fone=-1*one
#     leftm=np.concatenate((E,fone),axis=1)
#     result=leftm.dot(item_feature)
#     result=np.power(result,2)
#     result=result.sum(axis=1)
#
#
#
#     t1=time.time()
#     print(t1-t0)
#
#
# np.save('../code_ml/RW_adj_score/NoP_FGitem_adj.npy',user_adj_dict)
# print(user_adj_dict)






















