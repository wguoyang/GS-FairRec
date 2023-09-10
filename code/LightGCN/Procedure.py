
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from time import time
from tqdm import tqdm
import model
import multiprocessing
import pc_gender_train
import pc_age_train
import pc_occ_train
import copy
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2

def BPR_train_original(dataset, recommend_model, loss_class):
    Recmodel = recommend_model
    Recmodel.train()
    bpr= loss_class
    S, sam_time = utils.UniformSample_original(dataset)
    print(f"BPR[sample time][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    return f"[BPR[aver loss{aver_loss:.3e}]"

def MSE_train_original( train_loader,recommend_model, loss_class):
    Recmodel = recommend_model
    Recmodel.train()
    mseloss = loss_class
    for user_batch, rating_batch, item_batch in train_loader:
        user_batch = user_batch.cuda()
        rating_batch = rating_batch.cuda()
        item_batch = item_batch.cuda()
        mseloss.stageOne(user_batch,rating_batch,item_batch)

def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall),
            'precision':np.array(pre),
            'ndcg':np.array(ndcg)}

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def Test(dataset, Recmodel,multicore=0,):
    u_batch_size = world.config['test_u_batch_size']
    testDict= dataset.testDict
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    # results = {'precision': np.zeros(len(world.topks)),
    #            'recall': np.zeros(len(world.topks)),
    #            'ndcg': np.zeros(len(world.topks))}
    results={}
    with torch.no_grad():
        # users = list(testDict.keys())
        # test_items = list(set(dataset.testItem)) #set the testitem list only in test dataset
        # allitem_list = list(range(dataset.m_items))
        # outitems = list(set(allitem_list) - set(test_items))
        # auc_record = []
        # # ratings = []
        # total_batch = len(users) // u_batch_size + 1
        # allPos = dataset.getUserPosItems(users)
        # groundTrue = [testDict[u] for u in users]
        # users_gpu = torch.Tensor(users).long()
        # users_gpu = users_gpu.to(world.device)
        #
        # rating = Recmodel.getUsersRating(users_gpu)
        # #rating = rating.cpu()
        # exclude_index = []
        # exclude_items = []
        # for range_i, items in enumerate(allPos):
        #     exclude_index.extend([range_i] * len(items))
        #     exclude_items.extend(items)
        #     exclude_index.extend([range_i] * len(outitems))
        #     exclude_items.extend(outitems)
        # rating[exclude_index, exclude_items] = -(1<<10)
        # _, rating_K = torch.topk(rating, k=max_K)
        # rating = rating.cpu().numpy()
        # aucs = [
        #         utils.AUC(rating[i],
        #                   dataset,
        #                   test_data) for i, test_data in enumerate(groundTrue)
        #     ]
        # auc_record.extend(aucs)
        # del rating
        # x = [rating_K.cpu(), groundTrue]
        # result = test_one_batch(x)
        #
        # results['recall'] += result['recall']
        # results['precision'] += result['precision']
        # results['ndcg'] += result['ndcg']
        # results['recall'] /= float(len(users))
        # results['precision'] /= float(len(users))
        # results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        #rmse
        user_embed, item_embed = Recmodel.computer()
        user_embed = user_embed.cpu().detach().numpy()
        item_embed = item_embed.cpu().detach().numpy()
        #test_rating_dict=np.load('../data/ml1m/testing_ratings_dict.npy',allow_pickle=True)
        test_rating_dict, test_count = np.load('../../data/ml1m/data1t5/testing_ratings_dict.npy',
                                               allow_pickle=True)
        # test_rating_dict, test_count = np.load('../../data/lastfm/data1t5/testing_ratings_dict.npy',
        #                                        allow_pickle=True)
        pre_all = []
        label_all = []
        for pair_i in test_rating_dict:
            u_id, r_v, i_id = test_rating_dict[pair_i]
            r_v+=1
            pre_get = np.sum(user_embed[u_id] * item_embed[i_id])
            pre_all.append(pre_get)
            label_all.append(r_v)
        r_test = rmse(np.array(pre_all), np.array(label_all))
        res_test = round(np.mean(r_test), 4)
        results['RMSE']=res_test
        # #auc
        # auc_one, auc_res = pc_gender_train.clf_gender_all_pre('gcn_gender_auc', -1, copy.deepcopy(user_embed), 64)
        # results['gender_auc'] = round(auc_one, 4)
        # f1p_one,f1r_one,f1res_p,f1res_r= pc_age_train.clf_age_all_pre('gcn_age_f1',-1,copy.deepcopy(user_embed),64)
        #
        # f1micro_f1=(2*f1p_one*f1r_one)/(f1p_one+f1r_one)
        # results['age_f1'] = round(f1micro_f1, 4)
        # f1p_one,f1r_one,f1res_p,f1res_r= pc_occ_train.clf_occ_all_pre('gcn_occ_f1',-1,copy.deepcopy(user_embed),64)
        # # f1pone=np.mean(f1res_p)
        # # f1rone=np.mean(f1res_r)
        # f1micro_f1=(2*f1p_one*f1r_one)/(f1p_one+f1r_one)
        #
        #
        #
        # results['occ_f1'] = round(f1micro_f1, 4)
        # #
        print(results)
        return results