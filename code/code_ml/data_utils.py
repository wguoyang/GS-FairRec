# -- coding:UTF-8
import numpy as np 
# import pandas as pd 
import scipy.sparse as sp 
from scipy.sparse import csr_matrix
import torch.utils.data as data
import pdb
from torch.autograd import Variable
import torch
import math
import random


def RW_filter(userp):
    # 参数，用户邻居节点过滤的比例，物品邻居节点过滤的比例
    # 返回过滤掉的节点个数，以及过滤节点的id
    user_adj_score = np.load('RW_adj_score/FGuser_adj.npy', allow_pickle=True).item()
    #item_adj_score = np.load('RW_adj_score/item_adj.npy', allow_pickle=True).item()
    user_f_num = []
    user_f_id = {}
    for uid in range(6040):
        us = user_adj_score['user_' + str(uid)]
        f_num = int(len(us) * userp)
        if f_num < 1:
            f_num = 1
        user_f_num.append(f_num)
        dit = sorted(us.items(), key=lambda x: x[1])[:f_num]
        lin_id = []
        for it, sc in dit:
            lin_id.append(int(it[5:]))
        user_f_id[uid] = lin_id
    return user_f_num, user_f_id
def adj_mairx(train_set,ifRW):
    train_rating=train_set
    userid, itemid, rating = [], [], []
    for i in train_rating.keys():
        userid.append(train_rating[i][0])
        rating.append(train_rating[i][1])
        itemid.append(train_rating[i][2])
    # userid=train_set[:,0]
    # rating=train_set[:,1]
    rating=np.array(rating)
    one = np.ones(shape=(900188,), dtype=int)
    rating=rating+one
    rating=rating.astype(np.float32)
    # itemid=train_set[:,2]
    UserItemNet = csr_matrix((rating, (userid, itemid)),
                             shape=(6040, 3952))  # 创建一个稀疏矩阵

    adj=UserItemNet.todok()
    #过滤
    if ifRW==True:
        user_f_num, user_f_id = RW_filter(0.08)
        for uid in user_f_id.keys():
            for iid in user_f_id[uid]:
                adj[uid,iid] = 0
    rowsum = np.array(adj.sum(axis=1))
    d_inv = np.power(rowsum, -1).flatten()  # 开根号
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)  # 转为对角矩阵
    adj_mat=d_mat.dot(adj)
    adj_mat=adj_mat.tocsr()
    coo = adj_mat.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    final_adj=torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    return final_adj.coalesce().to('cuda')

def intersection_matrix(user_select):
    row,col,data=[],[],[]
    for uid in user_select.keys():
        k=1/(len(user_select[uid]))
        for iid in user_select[uid]:
            row.append(uid)
            col.append(iid)
            data.append(k)
    row=torch.Tensor(row).long()
    col=torch.Tensor(col).long()
    index=torch.stack([row,col])
    data=torch.FloatTensor(data)
    final_adj=torch.sparse.FloatTensor(index,data,torch.Size([6040,3952]))
    return final_adj.coalesce().to('cuda')

class ArrayData(data.Dataset):
    def __init__(self, train_raing_dict=None, is_training=None):
        super(ArrayData, self).__init__()

        self.train_raing_dict = train_raing_dict
        self.is_training = is_training
        (numx,numy)=train_raing_dict.shape
        self.data_set_count = numx
        # 构造这个类的对象时需要三个参数

    def __len__(self):  # 获取数据集长度
        return self.data_set_count  # return self.num_ng*len(self.train_dict)

    def __getitem__(self, idx):  # 获取给定索引的项目，包括特征用户，标签，物品
        features = self.train_raing_dict
        user = features[idx,0]
        label_r = np.array(features[idx,1])  # 创建为一个数组类型
        item = features[idx,2]
        return user, label_r.astype(np.float32), item  # float32  .astype(np.int)
#BPR类
class BPRData(data.Dataset):
    def __init__(self,train_raing_dict=None,is_training=None, data_set_count=0):
        super(BPRData, self).__init__()
 
        self.train_raing_dict = train_raing_dict 
        self.is_training = is_training
        self.data_set_count = data_set_count 
        #构造这个类的对象时需要三个参数

    def __len__(self):#获取数据集长度
        return self.data_set_count#return self.num_ng*len(self.train_dict)

    def __getitem__(self, idx):#获取给定索引的项目，包括特征用户，标签，物品
        features = self.train_raing_dict
        user = features[idx][0]
        label_r = np.array(features[idx][1]+1)#创建为一个数组类型
        item = features[idx][2]
        return user, label_r.astype(np.float32), item#float32  .astype(np.int)


class generate_adj():
    def __init__(self,training_user_set,training_item_set,user_num,item_num):
        self.training_user_set=training_user_set
        self.training_item_set=training_item_set
        self.user_num=user_num
        self.item_num=item_num 

    def readD(self,set_matrix,num_):
        user_d=[]
        for i in range(num_): 
            # len_set=1.0#/(len(set_matrix[i])+1)  
            len_set=1.0/(len(set_matrix[i])+1)  
            user_d.append(len_set)
        return user_d 
    #user-item  to user-item matrix and item-user matrix
    def readTrainSparseMatrix(self,set_matrix,is_user,u_d,i_d):
        user_items_matrix_i=[]
        user_items_matrix_v=[] 
        if is_user:
            d_i=u_d
            d_j=i_d
        else:
            d_i=i_d
            d_j=u_d
        for i in set_matrix: 
            len_set=len(set_matrix[i])#+1
            for pair_v in set_matrix[i]:
                # pdb.set_trace()
                r_v,j =pair_v  #r_v是打分,j电影id
                user_items_matrix_i.append([i,j])
                """
                形状如下
                i,j
                i,j
                i,j
                ...
                后面要转置为
                i,i,i,i,...
                j,j,j,j,...
                表示每个元素所在矩阵中的位置，以构造稀疏矩阵
                """
                # d_i_j=np.sqrt(d_i[i]*d_j[j])
                #1/sqrt((d_i+1)(d_j+1))
                user_items_matrix_v.append(r_v*1./len_set) #?
                # user_items_matrix_v.append(d_i_j)#(1./len_set) 
        user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
        user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
        return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v) #.t（）表示转置，构造稀疏矩阵
    
    def generate_pos(self): 
        u_d=self.readD(self.training_user_set,self.user_num)
        i_d=self.readD(self.training_item_set,self.item_num)
        #1/(d_i+1)
        d_i_train=u_d
        d_j_train=i_d
        sparse_u_i=self.readTrainSparseMatrix(self.training_user_set,True,u_d,i_d)
        sparse_i_u=self.readTrainSparseMatrix(self.training_item_set,False,u_d,i_d)
        #产生user_item_matrix,item_user_matrix,d_i_train,d_j_train
        return sparse_u_i,sparse_i_u,d_i_train,d_j_train
    
 





 
