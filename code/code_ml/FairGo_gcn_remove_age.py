# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys

os.environ["CUDA_VISIBLE_DEVICES"] ='0'
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
import utils

# seed=2022
# np.random.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
# torch.manual_seed(seed)
##movieLens-1M
user_num=6040#user_size
item_num=3952#item_size 
factor_num=64 #embedding维度
batch_size=2048*100#2048*100
top_k=20
num_negative_test_val=-1##all

dataset_base_path='../../data/ml1m'
saved_model_path='..'  

run_id="ga0"
print(run_id)
dataset='movieLens-1M'

training_user_set,training_item_set,training_set_count = np.load(dataset_base_path+'/datanpy/training_set.npy',allow_pickle=True)
testing_user_set,testing_item_set,testing_set_count = np.load(dataset_base_path+'/datanpy/testing_set.npy',allow_pickle=True)  
val_user_set,val_item_set,val_set_count = np.load(dataset_base_path+'/datanpy/val_set.npy',allow_pickle=True)    
user_rating_set_all,_,_ = np.load(dataset_base_path+'/datanpy/user_rating_set_all.npy',allow_pickle=True)


training_ratings_dict,train_dict_count = np.load(dataset_base_path+'/data1t5/training_ratings_dict.npy',allow_pickle=True)
testing_ratings_dict,test_dict_count = np.load(dataset_base_path+'/data1t5/testing_ratings_dict.npy',allow_pickle=True)

training_u_i_set,training_i_u_set = np.load(dataset_base_path+'/data1t5/training_adj_set.npy',allow_pickle=True)
# training_u_i_set_file=open('D:/gugedownload/NF-FairRec/NF-FairRec/code/code_ml/trainsave.txt','w+')
# training_u_i_set_file.write(str(training_u_i_set))
# training_u_i_set_file.close()

# users_emb_gcn = np.load('./LightGCN_embedding/final_version/user_embedding_110.npy',allow_pickle=True)
# #users_emb_gcn是一个矩阵6040行64列，也就是embedding向量是64维的
# items_emb_gcn = np.load('./LightGCN_embedding/final_version/item_embedding_110.npy',allow_pickle=True)
#items_emb_gcn是一个矩阵3952行64列
# users_emb_gcn = np.load('gcnModel/LGN_MLP_user_embedding_NF_0.12_36.npy', allow_pickle=True)
# items_emb_gcn = np.load('gcnModel/LGN_MLP_item_embedding_NF_0.12_36.npy', allow_pickle=True)
users_emb_gcn = np.load('gcnModel/user_emb_epoch79.npy', allow_pickle=True)
items_emb_gcn = np.load('gcnModel/item_emb_epoch79.npy', allow_pickle=True)
users_features=np.load(dataset_base_path+'/data1t5/users_features_3num.npy')
shiyanmiaoshu='去敏感信息，年龄，FairGo，加上RecFair'
f = open('ml1m_age_result.txt', 'a')
f.write(shiyanmiaoshu+'\n')
f.close()
users_features=np.load(dataset_base_path+'/data1t5/users_features_3num.npy')


class InforMax(nn.Module):
    def __init__(self, user_num, item_num, factor_num,users_features,gcn_user_embs,gcn_item_embs):
        #InforMax类需要的参数用户特征，GCN用户emb向量，物品emb向量
        super(InforMax, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        three module: LineGCN, AvgReadout, Discriminator（三个模型）
        """
        # self.gcn = filter_layer.AttributeLineGCN(user_num, item_num, factor_num,users_features,items_features)
        # pdb.set_trace() 
        self.users_features = torch.cuda.LongTensor(users_features)
        self.gcn_users_embedding0 = torch.cuda.FloatTensor(gcn_user_embs)
        self.gcn_items_embedding0 = torch.cuda.FloatTensor(gcn_item_embs)
        self.user_num = user_num
        
        self.sigm = nn.Sigmoid()#激活函数
        self.mse_loss=nn.MSELoss()#损失函数
        self.model_d1 = filter_layer.DisClfGender(factor_num,2,attribute='gender',use_cross_entropy=True)#创建性别属性判别器输入向量维度64，输出维度是2，
        self.model_d2 = filter_layer.DisClfAge(factor_num,7,attribute='age',use_cross_entropy=True)
        self.model_d3 = filter_layer.DisClfOcc(factor_num,21,attribute='occupation',use_cross_entropy=True)

        #分别对三个属性创建3个判别器
        self.model_f1 = filter_layer.AttributeFilter(factor_num, attribute='gender')
        self.model_f2 = filter_layer.AttributeFilter(factor_num, attribute='age')
        self.model_f3 = filter_layer.AttributeFilter(factor_num, attribute='occupation')

        # Adversarial ground truths 创建了两个全为1和全为0的张量，作用未知
        self.real_d = torch.ones(user_num, 1).cuda()#Variable(torch.Tensor(user_num).fill_(1.0), requires_grad=False)
        self.fake_d = torch.zeros(user_num, 1).cuda()#Variable(torch.Tensor(user_num).fill_(0.0), requires_grad=False)

    def forward(self, adj_pos,user_batch,rating_batch,item_batch):  #adj_pos是user_item_matrix,item_user_matrix,d_i_train,d_j_train
        #format of pos_seq or neg_seq:user_item_matrix,item_user_matrix,d_i_train,d_j_train
        adj_pos1 = copy.deepcopy(adj_pos)
        gcn_users_embedding0 = self.gcn_users_embedding0
        gcn_items_embedding0 = self.gcn_items_embedding0 

        # filter gender, age,occupation
        user_f1_tmp = self.model_f1(gcn_users_embedding0)#性别，生成过滤后的用户emb向量
        user_f2_tmp = self.model_f2(gcn_users_embedding0)#年龄
        user_f3_tmp = self.model_f3(gcn_users_embedding0)#职业
        #binary mask 
        #d_mask = [torch.randint(0,2,(1,)),torch.randint(0,2,(1,)),torch.randint(0,2,(1,))]
        d_mask=[0,1,0]
        d_mask = torch.cuda.FloatTensor(d_mask)#.cuda()
        sum_d_mask = d_mask[0]+d_mask[1]+d_mask[2] 
        while sum_d_mask <= 0:# ensure at least one filter
            d_mask = [torch.randint(0,2,(1,)),torch.randint(0,2,(1,)),torch.randint(0,2,(1,))] 
            d_mask = torch.cuda.FloatTensor(d_mask)
            sum_d_mask = d_mask[0]+d_mask[1]+d_mask[2]
        #以上四行代码确保d_mask中至少有一个为1

        user_f_tmp = (d_mask[0]*user_f1_tmp+d_mask[1]*user_f2_tmp+d_mask[2]*user_f3_tmp)/sum_d_mask
        #至少要有一个过滤器，对三个过滤后的向量进行聚合

        lables_gender = self.users_features[:,0]
        d_loss1 = self.model_d1(user_f_tmp,lables_gender)
        lables_age = self.users_features[:,1]
        d_loss2 = self.model_d2(user_f_tmp,lables_age)
        lables_occ = self.users_features[:,2]
        d_loss3 = self.model_d3(user_f_tmp,lables_occ)
        #计算三个判别器的loss，这里判别器的输入是过滤后的userembedding向量
        # pdb.set_trace()

        # #local attribute
        item_f1_tmp = self.model_f1(gcn_items_embedding0)
        item_f2_tmp = self.model_f2(gcn_items_embedding0)
        item_f3_tmp = self.model_f3(gcn_items_embedding0)
        #生成含有图结构的用户表示
        users_f1_local = torch.sparse.mm(adj_pos1, item_f1_tmp)#稀疏矩阵乘法，所得到的矩阵是6040*64的，每一行是每个用户的一阶聚合表示
        # print(item_f1_tmp)
        # print(users_f1_local)
        users_f2_local = torch.sparse.mm(adj_pos1, item_f2_tmp)
        users_f3_local = torch.sparse.mm(adj_pos1, item_f3_tmp)
        user_f_local_tmp = (d_mask[0]*users_f1_local+d_mask[1]*users_f2_local+d_mask[2]*users_f3_local)/sum_d_mask
        #user_f_local_tmp也会作为判别器输入，计算一个local loss
        # lables_gender = self.users_features[:,0]
        # lables_age = self.users_features[:,1]
        # lables_occ = self.users_features[:,2]
        d_loss1_local = self.model_d1(user_f_local_tmp,lables_gender)
        d_loss2_local = self.model_d2(user_f_local_tmp,lables_age)
        d_loss3_local = self.model_d3(user_f_local_tmp,lables_occ)

        w_f=[1,2,1]#[5,2.5,0.5]#[1,1,1]#[5,2.5,0.5]#[1,1,1]#[5,2.5,0.5]#[1,1,1]#[5,2.5,0.5]#[1,1,1]#[5,2.5,0.5]#[3,1.5,0.5]#[5,2.5,0.5]#[1,1,1]#[5,2.5,0.5]
        #????
        d_loss = (d_mask[0]*d_loss1*w_f[0]+ d_mask[1]*d_loss2*w_f[1] + d_mask[2]*d_loss3*w_f[2])#/sum_d_mask
        d_loss_local = (d_mask[0]*d_loss1_local*w_f[0]+ d_mask[1]*d_loss2_local*w_f[1] + d_mask[2]*d_loss3_local*w_f[2])#/sum_d_mask

        #L_R preference prediction loss.
        user_person_f = user_f2_tmp
        item_person_f = item_f2_tmp#gcn_items_embedding0#item_f2_tmp

        user_b = F.embedding(user_batch,user_person_f)#将每个embedding与用户对应起来
        item_b = F.embedding(item_batch,item_person_f)
        prediction = (user_b * item_b).sum(dim=-1)#用过滤之后的embedding向量做内积得到预测分数
        loss_part = self.mse_loss(prediction,rating_batch)#计算预测的loss
        l2_regulization = 0.01*(user_b**2+item_b**2).sum(dim=-1)
        # loss_part= -((prediction_i - prediction_j).sigmoid().log().mean())
        loss_p_square=loss_part+l2_regulization.mean()#正则化后的评分预测loss，过滤器想要最小化这个loss

        d_loss_all= 1*(d_loss+0.5*d_loss_local)#+1*d_loss_local #+1*d_loss1_local.cpu().numpy()
        #d_loss_all=d_loss
        g_loss_all= 10*loss_p_square #- 1*d_loss_all#评分预测loss
        g_d_loss_all = - 1*d_loss_all
        d_g_loss = [d_loss_all,g_loss_all,g_d_loss_all]

        return d_g_loss
    # Detach the return variables
    def embed(self, adj_pos):
        # h_pos: cat gcn_users_embedding and gcn_items_embedding, dim =0 
        fliter_u_emb1 = self.model_f1(self.gcn_users_embedding0)
        fliter_u_emb2 = self.model_f2(self.gcn_users_embedding0)
        fliter_u_emb3 = self.model_f3(self.gcn_users_embedding0)
        fliter_i_emb1 = self.model_f1(self.gcn_items_embedding0)
        fliter_i_emb2 = self.model_f2(self.gcn_items_embedding0)
        fliter_i_emb3 = self.model_f3(self.gcn_items_embedding0) 
        # fliter_i_emb = self.gcn_items_embedding0
        return fliter_u_emb1.detach(),fliter_u_emb2.detach(),fliter_u_emb3.detach(),fliter_i_emb1.detach(),fliter_i_emb2.detach(),fliter_i_emb3.detach()
        #如果A网络的输出被喂给B网络作为输入， 如果我们希望在梯度反传的时候只更新B中参数的值，而不更新A中的参数值，这时候就可以使用detach()

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))
 
###
# g_adj= data_utils.generate_adj(training_u_i_set,training_i_u_set,user_num,item_num)
# pos_adj=g_adj.generate_pos() #产生user_item_matrix,item_user_matrix,d_i_train,d_j_train
#print(pos_adj[0])展示稀疏矩阵是展示行列索引和其对应的值，而不是以整个矩阵形式展示
###
pos_adj=data_utils.adj_mairx(training_ratings_dict,ifRW=False)
###加载数据集
# user_select=np.load('./select/user_select_0.5_age.npy',allow_pickle=True)
# user_select=user_select.item()
# pos_adj=data_utils.intersection_matrix(user_select)
train_dataset = data_utils.BPRData(
        train_raing_dict=training_ratings_dict,is_training=True, data_set_count=train_dict_count)
train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True,num_workers=0)#num_workers=2
  
testing_dataset_loss = data_utils.BPRData(
        train_raing_dict=testing_ratings_dict,is_training=False, data_set_count=test_dict_count)
testing_loader_loss = DataLoader(testing_dataset_loss,
        batch_size=test_dict_count, shuffle=False, num_workers=0)
FairMetric=utils.FairAndPrivacy()
###


######################################################## TRAINING #####################################

all_nodes_num=user_num+item_num #总结点数
print('--------training processing-------')
count, best_hr = 0, 0
 
model = InforMax(user_num, item_num, factor_num,users_features,users_emb_gcn,items_emb_gcn)
model=model.to('cuda') 

###优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#, betas=(0.5, 0.99)) #优化器，针对全部参数
# d1_optimizer = torch.optim.Adam(model.model_d1.parameters(), lr=0.005)
# f1_optimizer = torch.optim.Adam(model.model_f1.parameters(), lr=0.005) 
# gcn_optimizer = torch.optim.Adam(model.gcn.parameters(), lr=0.005)
f_optimizer = torch.optim.Adam(
                            list(model.model_f2.parameters()) ,lr=0.001)#针对过滤器参数
d_optimizer = torch.optim.Adam(
                            list(model.model_d2.parameters()) ,lr=0.001)#针对判别器参数
###

###计算rmse的函数
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
###

if __name__ == '__main__':
    for epoch in range(150):#训练100轮
        model.train()#模型调为训练模式
        start_time = time.time()
        print('train data is  end')

        loss_current = [[],[],[],[]]#用来保存当前loss
        #0过滤器预测loss，1用于优化过滤器的判别器loss，2判别器loss


        ###训练判别器
        for user_batch, rating_batch, item_batch in train_loader:
            user_batch = user_batch.cuda()
            rating_batch = rating_batch.cuda()
            item_batch = item_batch.cuda()
            d_g_l_get =  model(copy.deepcopy(pos_adj),user_batch,rating_batch,item_batch)
            d_l,f_l,_ = d_g_l_get
            # d_loss_all = 1 * (d_loss + 1 * d_loss_local)  # +1*d_loss_local #+1*d_loss1_local.cpu().numpy()
            # g_loss_all = 10 * loss_p_square  # - 1*d_loss_all#评分预测loss
            # g_d_loss_all = - 1 * d_loss_all
            # d_g_loss = [d_loss_all, g_loss_all, g_d_loss_all]
            loss_current[2].append(d_l.item())
            d_optimizer.zero_grad()#梯度置0
            d_l.backward()#反向传播
            d_optimizer.step()#优化参数
        ###

        ###训练过滤器
        for user_batch, rating_batch, item_batch in train_loader:
            user_batch = user_batch.cuda()
            rating_batch = rating_batch.cuda()
            item_batch = item_batch.cuda()
            d_g_l_get = model(copy.deepcopy(pos_adj),user_batch,rating_batch,item_batch)
            # d_g_l_get = model(copy.deepcopy(pos_adj),copy.deepcopy(pos_adj),user,item_i, item_j)
            _,f_l,d_l = d_g_l_get
            loss_current[0].append(f_l.item())
            # loss_current[1].append(d_l.item())
            f_optimizer.zero_grad()
            f_l.backward()
            f_optimizer.step()
            # continue
        ###

        ###优化过滤器使得判别器loss增大

        d_g_l_get =  model(copy.deepcopy(pos_adj),user_batch,rating_batch,item_batch)
        _,f_l,d_l = d_g_l_get
        loss_current[1].append(d_l.item())
        f_optimizer.zero_grad()
        d_l.backward()
        f_optimizer.step()
        ###

        ###这部分代码是为了显示一些信息
        loss_current=np.array(loss_current,dtype=object)
        elapsed_time = time.time() - start_time
        # pdb.set_trace()
        train_loss_f = round(np.mean(loss_current[0]),4)#算一个平均之后四舍五入到小数点后四位
        train_loss_f_d = round(np.mean(loss_current[1]),4)#
        train_loss_d=round(np.mean(loss_current[2]),4)#
        str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))#+' train loss:'+str(train_loss)+'='+str(train_loss_part)+'+'

        str_d_g_str=' loss'
        # str_d_g_str+=' f:'+str(train_loss_f)+'='+str(train_loss_f_g)+' - '+str(train_loss_f_d)
        str_d_g_str+=' f:'+str(train_loss_f)+'fd:'+str(train_loss_f_d)
        str_d_g_str+='\td:'+str(train_loss_d)#
        str_print_train +=str_d_g_str#'  d_1:'+str()
        print(run_id+'--train--',elapsed_time)
        print(str_print_train)#显示loss
        ###

        # result_file.write(str_print_train)#保存在一个文件里
        # result_file.write('\n')
        # result_file.flush()

        model.eval()#模型调为测试模式

        f1_users_embedding,f2_users_embedding,f3_users_embedding,f1_i_emb,f2_i_emb,f3_i_emb= model.embed(copy.deepcopy(pos_adj))
        user_e_f2 = f2_users_embedding.cpu().numpy()
        item_e_f2 = f2_i_emb.cpu().numpy()

        user_e = user_e_f2
        item_e = item_e_f2#items_emb_gcn
        str_print_evl=''#'epoch:'+str(epoch)
        pre_all = []#保存预测打分
        label_all = []#保存实际打分
        for pair_i in testing_ratings_dict:
            u_id, r_v, i_id = testing_ratings_dict[pair_i]#[用户id，评分，电影id]
            r_v+=1
            pre_get = np.sum(user_e[u_id]*item_e[i_id])
            pre_all.append(pre_get)
            label_all.append(r_v)
        r_test=rmse(np.array(pre_all),np.array(label_all))#计算预测和实际打分的rmse
        res_test=round(np.mean(r_test),4)#保留四位小数
        str_print_evl+="\trmse:"+str(res_test)
        precision_Att,recall_Att = FairMetric.get_Att_precision(user_e, item_e, 'age')
        precision_Att = round(precision_Att, 4)
        str_print_evl += "\tLACC_precision:" + str(precision_Att)
        f1micro_f1 = (2 * precision_Att * recall_Att) / (precision_Att + recall_Att)
        str_print_evl += "\tLACC_f1:" + str(round(f1micro_f1, 4))
        #分类器的F1
        #print(user_e)
        # f1p_one, f1r_one, f1res_p, f1res_r = pc_age_train.clf_age_all_pre(run_id, epoch, user_e, factor_num)
        # #f1pone = np.mean(f1res_p)
        # #print(f1pone)
        # #f1rone = np.mean(f1res_r)
        # f1micro_f1 = (2 * f1p_one * f1r_one) / (f1p_one + f1r_one)
        # str_f1f1_res = ' age f1:' + str(round(f1micro_f1, 4)) + '\t'
        #str_print_evl += str_f1f1_res
        print(str_print_evl)
        f = open('ml1m_age_result.txt', 'a')
        f.write(str_print_train+' '+str_print_evl + '\n')
        f.close()
