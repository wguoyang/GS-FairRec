# -- coding:UTF-8
import numpy as np 
# import pandas as pd 
import scipy.sparse as sp 
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.utils.data as data
import pdb
from torch.autograd import Variable
import torch
import math
import random

class BPRData(data.Dataset):
    def __init__(self,train_raing_dict=None,is_training=None, data_set_count=0):
        super(BPRData, self).__init__()
 
        self.train_raing_dict = train_raing_dict 
        self.is_training = is_training
        self.data_set_count = data_set_count 
      
    def __len__(self):
        return self.data_set_count#return self.num_ng*len(self.train_dict)

    def __getitem__(self, idx):
        features = self.train_raing_dict
        user = features[idx][0]
        label_r = np.array(features[idx][1])
        item = features[idx][2]
        return user, label_r.astype(np.float32), item#float32  .astype(np.int)

class FairAndPrivacy:
    def __init__(self,topk=5):
        item_feature=np.load('../../data/lastfm/data1t5/item_feature.npy',allow_pickle=True)
        #user_feature=np.load('../../data/ml1m/data1t5/users_features_list.npy',allow_pickle=True)
        user_feature=np.load('../../data/lastfm/data1t5/users_features_2num.npy',allow_pickle=True)
        user_sample=np.load('../../data/lastfm/data1t5/RecFair_user_sample.npy',allow_pickle=True)
        item_sample = np.load('../../data/lastfm/data1t5/RecFair_item_sample.npy', allow_pickle=True)
        self.item_feature=item_feature[item_sample,:]
        #print(self.item_feature)
        # for iid in range(itemnum):
        #     if self.item_feature[iid,0]>0:
        #         self.item_feature[iid,0]=self.item_feature[iid,0]*0.3
        #     else:
        #         self.item_feature[iid, 0] = self.item_feature[iid, 0] * 0.7
        #print(self.item_feature)
        self.user_feature=user_feature[user_sample,:]
        self.topk=topk
        self.usernum=len(user_sample)
        self.itemnum=len(item_sample)
        self.user_sample=user_sample
        self.item_sample=item_sample
    def get_Att_precision(self,user_embedding,item_embedding,Attribute):
        user_emb=user_embedding[self.user_sample,:]
        item_emb=item_embedding[self.item_sample,:]
        reclist=self.get_reclist(user_emb,item_emb)
        reclist=reclist.cpu().numpy()
        #print(reclist)
        if Attribute=='gender':
            predict_Att=[]
            for uid in range(self.usernum):
                rl=reclist[uid]
                recitem=self.item_feature[rl,:2]
                #weight=[10,9,8,7,6,5,4,3,2,1]
                #weight=np.array(weight).reshape((1,self.topk))
                #colsum=weight.dot(recitem)
                colsum=recitem.sum(axis=0)
                if colsum[0]>0:
                    predict_Att.append(1)
                else:
                    predict_Att.append(0)
            true_Att=self.user_feature[:,0]
            predict_Att=np.array(predict_Att)
        if Attribute=='age':
            predict_Att = []
            for uid in range(self.usernum):
                rl = reclist[uid]
                recitem = self.item_feature[rl,2:]
                # weight=[10,9,8,7,6,5,4,3,2,1]
                # weight=np.array(weight).reshape((1,self.topk))
                # colsum=weight.dot(recitem)
                colsum = recitem.sum(axis=0)
                pre =0
                for i in colsum:
                    if i>=0:
                        break
                    else:
                        pre+=1
                predict_Att.append(pre)
            true_Att = self.user_feature[:, 1]
            predict_Att = np.array(predict_Att)
        # if Attribute=='occ':
            # predict_Att = []
            # for uid in range(self.usernum):
            #     rl = reclist[uid]
            #     recitem = self.item_feature[rl,9:]
            #     # weight=[10,9,8,7,6,5,4,3,2,1]
            #     # weight=np.array(weight).reshape((1,self.topk))
            #     # colsum=weight.dot(recitem)
            #     colsum = recitem.sum(axis=0)
            #     pre =0
            #     for i in colsum:
            #         if i>=0:
            #             break
            #         else:
            #             pre+=1
            #     predict_Att.append(pre)
            # true_Att = self.user_feature[:, 2]
            # predict_Att = np.array(predict_Att)
        #print(predict_Att)
        return precision_score(true_Att,predict_Att,average='micro')
    def get_reclist(self,user_embedding,item_embedding):
        item_embedding=item_embedding.T
        rating=user_embedding.dot(item_embedding)
        rating=torch.cuda.FloatTensor(rating)
        _,reclist=rating.topk(self.topk,dim=1)#reclist是(usernum,topk)
        return reclist


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
                r_v,j =pair_v
                user_items_matrix_i.append([i,j])
                # d_i_j=np.sqrt(d_i[i]*d_j[j])
                #1/sqrt((d_i+1)(d_j+1))
                user_items_matrix_v.append(r_v*1./len_set) 
                # user_items_matrix_v.append(d_i_j)#(1./len_set) 
        user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
        user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
        return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v) 
    
    def generate_pos(self): 
        u_d=self.readD(self.training_user_set,self.user_num)
        i_d=self.readD(self.training_item_set,self.item_num)
        #1/(d_i+1)
        d_i_train=u_d
        d_j_train=i_d
        sparse_u_i=self.readTrainSparseMatrix(self.training_user_set,True,u_d,i_d)
        sparse_i_u=self.readTrainSparseMatrix(self.training_item_set,False,u_d,i_d)
        #user_item_matrix,item_user_matrix,d_i_train,d_j_train  
        return sparse_u_i,sparse_i_u,d_i_train,d_j_train
    
 





 
