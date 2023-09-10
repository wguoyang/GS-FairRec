# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys


# os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [1]))

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd 
import torch.utils.data as data

from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
import pdb
import copy 
from collections import defaultdict
import time
import data_utils 
from shutil import copyfile
import pickle 
import filter_layer


class Classifier(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super(Classifier, self).__init__()
        """ 
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        three module: LineGCN, AvgReadout, Discriminator
        """     
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim/4), bias=True), 
            nn.LeakyReLU(0.2,inplace=True), 
            nn.Linear(int(self.embed_dim/4), int(self.embed_dim/8), bias=True), 
            nn.LeakyReLU(0.2,inplace=True), 
            nn.Linear(int(self.embed_dim /8),int(self.embed_dim /16), bias=True), 
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(int(self.embed_dim /16), self.out_dim , bias=True), 
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(self.out_dim, self.out_dim , bias=True),
            nn.Sigmoid()
            
            )
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
    def forward(self, emb0,label0):
        scores = self.net(emb0)
        # pdb.set_trace()
        # outputs = F.log_softmax(scores, dim=1)
        # scores =scores.view(-1)
        loss = self.criterion(scores, label0)
        # pdb.set_trace()
        return loss 

    def prediction(self, emb0): 
        scores = self.net(emb0) 
        return scores.detach()

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))


class ClassifyData(data.Dataset):
    def __init__(self,data_set_count=0, train_data=None, is_training=None,embed_dim=0):
        super(ClassifyData, self).__init__() 
        self.is_training = is_training
        self.data_set_count = data_set_count
        self.features_fill = train_data 
        self.embed_dim = embed_dim
    def __len__(self):  
        return self.data_set_count#return self.num_ng*len(self.train_dict)
          
    def __getitem__(self, idx):
        features = self.features_fill
        feature_user = features[idx][:self.embed_dim ]
        label_user = features[idx][self.embed_dim:]
        label_user = label_user.astype(np.float32)
        return feature_user, label_user


def clf_gender_all_pre(model_id,epoch_run,users_embs,factor_num):
    ##movieLens-1M
    # user_num=6040#user_size
    # item_num=3952#item_size  
    # factor_num=64
    batch_size=2048*128
    dataset_base_path='../../data/lastfm'

    epoch_id='clf_gender/'+str(epoch_run) 
    print(model_id,epoch_id)
    dataset='lastfm'

    ####模型保存相关
    # path_save_base='../ClfLog/'+dataset+'/filter_lineargcn'+model_id+'/'+epoch_id
    # if (os.path.exists(path_save_base)):
    #     noprint=1#print('has results save path')
    # else:
    #     noprint=1#print('no results save path')
    #     os.makedirs(path_save_base)
    # result_file=open(path_save_base+'/results_age_classification.txt','w+')
    # result_file_hr_ndcg=open(path_save_base+'/results_age_auc.txt','w+')
    # copyfile('./pc_gender_train.py', path_save_base+'/pc_gender_train.py')
    #
    # path_save_model_base='../Model/'+dataset+'/filter_lineargcn'+model_id+'/'+epoch_id
    # if (os.path.exists(path_save_model_base)):
    #     print('has model save path')
    # else:
    #     print('no model save path')
    #     os.makedirs(path_save_model_base)
    
    # users_features = np.load(dataset_base_path+'/data1t5/users_features_list.npy',allow_pickle=True)
    users_features = np.load(dataset_base_path+'/data1t5/users_features.npy',allow_pickle=True)

    users_features=users_features.astype(np.float32)
    users_features_age_oh=users_features[:,:2]
    users_features_age=[np.where(r==1)[0][0] for r in users_features_age_oh]
    users_features_age=np.array(users_features_age).astype(np.float32)
    users_features_age = (users_features_age).reshape(-1,1)

    users_embs_cat_att = np.concatenate((users_embs, users_features_age), axis=-1)
    np.random.shuffle(users_embs_cat_att)
    #359,347=2.5W+7W+
    train_data_all = users_embs_cat_att[:250000][:]
    training_count=len(train_data_all)
    test_data_all = users_embs_cat_att[250000:320000][:]
    testing_count=len(test_data_all)
    val_data_all = users_embs_cat_att[320000:][:]
    val_count=len(val_data_all)
    
    # PATH_model_train_data=path_save_model_base+'/train_data_all.npy'
    # np.save(PATH_model_train_data,train_data_all)
    # PATH_model_test_data=path_save_model_base+'/test_data_all.npy'
    # np.save(PATH_model_test_data,test_data_all)
    # PATH_model_test_data=path_save_model_base+'/val_data_all.npy'
    # np.save(PATH_model_test_data,val_data_all)

    train_dataset = ClassifyData(
            data_set_count=training_count, train_data=train_data_all,is_training = True,embed_dim=factor_num)
    train_loader = DataLoader(train_dataset,
            batch_size=training_count, shuffle=True, num_workers=8)

    testing_dataset_loss = ClassifyData(
            data_set_count=testing_count, train_data=test_data_all,is_training = False,embed_dim=factor_num)
    testing_loader = DataLoader(testing_dataset_loss,
            batch_size=testing_count, shuffle=False, num_workers=4)

    val_dataset_loss = ClassifyData(
            data_set_count=val_count, train_data=val_data_all,is_training = False,embed_dim=factor_num)
    val_loader = DataLoader(val_dataset_loss,
            batch_size=val_count, shuffle=False, num_workers=4)
 
    ######################################################## TRAINING #####################################
    # print('--------training processing-------')
    count, best_hr = 0, 0 
    model = Classifier(embed_dim=factor_num, out_dim=1)
    model=model.to('cuda')  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#, betas=(0.5, 0.99))  

    # PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    # #torch.save(model.state_dict(), PATH_model) 
    # model.load_state_dict(torch.load(PATH_model)) 
    # model.eval()
    current_loss=0
    flag_stop=-1
    res_center=''
    res_auc=[]
    for epoch in range(230):
        # print('pc_gender_epoch',epoch)
        model.train()  
        start_time = time.time() 
        # print('train data of ng_sample is  end')
        train_loss_sum=[]
        train_loss_bpr=[] 
        val_loss_sum=[] 
        for user_features, user_labels in train_loader:
            # pdb.set_trace()
            user_features = user_features.cuda()
            user_labels = user_labels.cuda() 
            
            loss_get = model(user_features,user_labels)  
            optimizer.zero_grad()
            loss_get.backward()
            optimizer.step()  
            count += 1
            train_loss_sum.append(loss_get.item()) 

        elapsed_time = time.time() - start_time
        train_loss=round(np.mean(train_loss_sum),4)# 
        str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))+' train loss:'+str(train_loss) 

        for user_features, user_labels in val_loader:
            # pdb.set_trace()
            user_features = user_features.cuda()
            user_labels = user_labels.cuda() 
            loss_get = model(user_features,user_labels)
            val_loss_sum.append(loss_get.item()) 
        val_loss=round(np.mean(val_loss_sum),4)# 
        str_print_train+='\t val loss:'+str(train_loss) 

        if val_loss>current_loss:
            flag_stop+=1
        current_loss=val_loss
        
        # if epoch>0:
        #     PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
        #     torch.save(model.state_dict(), PATH_model)
        #
        # result_file.write(str_print_train)
        # result_file.write('\n')
        # result_file.flush()

        model.eval()
        auc_test_all=[]
        for user_features, user_labels in testing_loader:
            user_features = user_features.cuda()
            user_labels = user_labels.numpy()
            get_scores = model.prediction(user_features)
            pre_scores=get_scores.cpu().numpy()
            # y = user_labels#np.max(user_labels,-1)
            # pred = pre_scores#np.max(pre_scores,-1) 
            y = (user_labels).reshape(-1) 
            pred_auc = pre_scores#np.max(pre_scores,axis=-1)
            fpr, tpr, thresholds = metrics.roc_curve(y, pred_auc, pos_label=1)
            auc_test = metrics.auc(fpr, tpr)

            auc_one=round(auc_test,4) 
            if auc_one<0.5:
                auc_one=1-auc_one
            str_f1='gender ,auc:'+str(round(auc_test,4))#+'  f1_micro:'+str(round(f1_micro,4))#+' auc:'+str(round(auc_test,4))
            if flag_stop>=2:
                # print("epoch:"+str(epoch)+str_f1)
                # res_center+=str(round(f1_micro,4))+' '
                res_auc.append(auc_one)
                # res_r.append(r_one)

        str_print_evl=str_print_train+" epoch:"+str(epoch)+str_f1 
        # result_file_hr_ndcg.write(str_print_evl)
        # result_file_hr_ndcg.write('\n')
        # result_file_hr_ndcg.flush()
        if flag_stop==4:
            return auc_one,res_auc
    return auc_one,res_auc






