#remove some unnecessary codes
import os
os.environ["CUDA_VISIBLE_DEVICES"] ='1'#选择显卡
import world
from torch import nn
from world import cprint
import utils
from model import LightGCN
import dataloader
from torch.utils.data import DataLoader
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
import pc_gender_train
import copy
import pc_age_train
import pc_occ_train
import matplotlib.pyplot as plt
from os.path import join
utils.set_seed(2022)
#print(torch.cuda.is_available())
dataset = dataloader.Loader(path="../data/"+world.dataset)
Recmodel = LightGCN(world.config, dataset) #initialize MODEL
Recmodel = Recmodel.to(world.device) #put model into GPU
bpr = utils.BPRLoss(Recmodel, world.config) #initialize Optim
mse=utils.MSELoss(Recmodel,world.config)
print(world.config['lightGCN_n_layers'])
ndcg = 0
best_result = {}
batch_size=2048#ml1m=2048,lastfm=2048*128
#换数据集需要调整
train_set,train_count=np.load('../../data/ml1m/data1t5/training_ratings_dict.npy',allow_pickle=True)
# test_set=np.load('../data/ml1m/testing_ratings_dict.npy',allow_pickle=True)
train_set=utils.DictData(train_raing_dict=train_set,is_training=True,data_set_count=train_count)
# test_set=utils.MSEData(train_raing_dict=test_set,is_training=False)
train_loader = DataLoader(train_set,
                          batch_size=batch_size, shuffle=True, num_workers=4)#lastfm=28,ml1m=8
# test_set,test_count=np.load('../../data/lastfm/data1t5/testing_ratings_dict.npy`',allow_pickle=True)
# train_set=utils.DictData(train_raing_dict=test_set,is_training=False,data_set_count=test_count)
# test_loader = DataLoader(test_set,
#         batch_size=batch_size, shuffle=False, num_workers=0)
# shiyanmiaoshu='这次在LightGCN上进行改动，数据集是lastfm,拉普拉斯矩阵是01，没有自回路，每层embedding采用MLP聚合,采用节点过滤，p=0.20,lr=0.0001'
# f = open('lightGCN_lastfm_result1.txt', 'a')
# f.write(shiyanmiaoshu+'\n')
# f.close()
shiyanmiaoshu='使用多层感知机聚合embedding，跑ml1m，lr=0.001，NF=MF，p=0.08，保存embedding'
f = open('lightGCN_ml1m_layer3_result1.txt', 'a')
f.write(shiyanmiaoshu+'\n')
f.close()
x,rmse,auc,af1,of1=[],[],[],[],[]
s_user_e,s_item_e=[],[]
tagrmse=100
for epoch in range(50):
    print('======================')
    print(f'EPOCH[{epoch}/{world.TRAIN_epochs}]')
    start = time.time()
    x.append(epoch)
    result = Procedure.Test(dataset, Recmodel, world.config['multicore'])
    rmse.append(result['RMSE'])
    # if result['RMSE']<=tagrmse:
    #     tagrmse=result['RMSE']
    # else:
    #     break
    # user_embed, item_embed = Recmodel.computer()
    # user_embed = user_embed.cpu().detach().numpy()
    # item_embed = item_embed.cpu().detach().numpy()
    # s_user_e=user_embed
    # s_item_e=item_embed
    if epoch==36:
        user_embed, item_embed = Recmodel.computer()
        user_embed = user_embed.cpu().detach().numpy()
        item_embed = item_embed.cpu().detach().numpy()
        np.save('../code_ml/gcnModel/LGN_MLP_user_embedding_NF_0.08_' + str(epoch) + '.npy', user_embed)
        np.save('../code_ml/gcnModel/LGN_MLP_item_embedding_NF_0.08_' + str(epoch) + '.npy', item_embed)
        break
    #     if world.config['ifRW'] == True:
    #         np.save('../code_ml/LightGCN_embedding/final_version/user_embedding_UserCF_0.20_' + str(epoch) + '.npy', user_embed)
    #         np.save('../code_ml/LightGCN_embedding/final_version/item_embedding_UserCF_0.20_' + str(epoch) + '.npy', item_embed)
    #     else:
    #         np.save('../code_ml/LightGCN_embedding/final_version/user_embedding_' + str(epoch) + '.npy', user_embed)
    #         np.save('../code_ml/LightGCN_embedding/final_version/item_embedding_' + str(epoch) + '.npy', item_embed)
    #     auc_one, auc_res = pc_gender_train.clf_gender_all_pre('gcn_gender_auc', -1, copy.deepcopy(user_embed), 64)
    #     print('gender auc' + str(round(np.mean(auc_res), 4)))
    #     f1p_one, f1r_one, f1res_p, f1res_r = pc_age_train.clf_age_all_pre('gcn_occ_f1', -1, copy.deepcopy(user_embed), 64)
    #     f1pone = np.mean(f1res_p)
    #     f1rone = np.mean(f1res_r)
    #     f1micro_f1 = (2 * f1pone * f1rone) / (f1pone + f1rone)
    #     print('age f1' + str(round(f1micro_f1, 4)))
    #     f1p_one,f1r_one,f1res_p,f1res_r= pc_occ_train.clf_occ_all_pre('gcn_occ_f1',-1,copy.deepcopy(user_embed),64)
    #     f1pone=np.mean(f1res_p)
    #     f1rone=np.mean(f1res_r)
    #     f1micro_f1=(2*f1pone*f1rone)/(f1pone+f1rone)
    #     print('occ f1' + str(round(f1micro_f1, 4)))

    #f = open('lightGCN_{}_layer{}_result1.txt'.format(world.dataset, world.config['lightGCN_n_layers']), 'a')
    f = open('lightGCN_ml1m_layer3_result1.txt', 'a')
    f.write('epoch = {} \n'.format(epoch))
    f.write('result = {} \n'.format(result))
    f.close()
    if epoch %10 == 0:
        cprint("[TEST]")
        # #result = Procedure.Test(dataset, Recmodel, world.config['multicore'])
        # x.append(epoch)
        # result = Procedure.Test(dataset, Recmodel, world.config['multicore'])
        # rmse.append(result['RMSE'])






        # user_emb, item_emb = Recmodel.computer()
        # user_emb = user_emb.cpu().detach().numpy()
        # auc_one, auc_res = pc_gender_train.clf_gender_all_pre('gcn_gender_auc', -1, copy.deepcopy(user_emb), 64)
        # print('gender auc'+str(round(np.mean(auc_res), 4)))
        # f1p_one, f1r_one, f1res_p, f1res_r = pc_age_train.clf_age_all_pre('gcn_occ_f1', -1, copy.deepcopy(user_emb), 64)
        # f1pone = np.mean(f1res_p)
        # f1rone = np.mean(f1res_r)
        # f1micro_f1 = (2 * f1pone * f1rone) / (f1pone + f1rone)
        # print('age f1' + str(round(f1micro_f1, 4)))


        # f1p_one,f1r_one,f1res_p,f1res_r= pc_occ_train.clf_occ_all_pre('gcn_occ_f1',-1,copy.deepcopy(user_emb),64)
        # f1pone=np.mean(f1res_p)
        # f1rone=np.mean(f1res_r)
        # f1micro_f1=(2*f1pone*f1rone)/(f1pone+f1rone)
        # print('occ f1' + str(round(f1micro_f1, 4)))
    #output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr)
    output_information = Procedure.MSE_train_original(train_loader,Recmodel, mse)
    print(f"[TOTAL TIME] {time.time() - start}")
# np.save('../code_lastfm/gcnModel/L_eui_user_embedding_NF_0.12_' + str(epoch) + '.npy', s_user_e)
# np.save('../code_lastfm/gcnModel/L_eui_item_embedding_NF_0.12_' + str(epoch) + '.npy', s_item_e)
# preuser_embedding,preitem_embedding=Recmodel.getpretrainembedding()
# np.save('../code_lastfm/gcnModel/pretrain_user_embedding_' + str(epoch) + '.npy', preuser_embedding)
# np.save('../code_lastfm/gcnModel/pretrain_item_embedding_' + str(epoch) + '.npy', preitem_embedding)
plt.plot(x,rmse,linestyle='--',marker='*',color='r')
plt.show()
# f = open('lightGCN_{}_layer{}_result.txt'.format(world.dataset, world.config['lightGCN_n_layers']), 'a')
# f.write('best_epoch = {} \n'.format(best_epoch))#250轮最好
# f.write('best_result = {} \n'.format(best_result))
#f.close()