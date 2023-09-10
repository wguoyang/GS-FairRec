

import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as F
from time import time
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score


class MSEData(data.Dataset):
    def __init__(self, train_raing_dict=None, is_training=None):
        super(MSEData, self).__init__()

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

class BPRLoss:
    def __init__(self, recmodel, config):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss *self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

class MSELoss:
    def __init__(self,recmodel,config):
        self.model=recmodel
        self.lr=config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        self.mse_loss = nn.MSELoss().cuda()

    def stageOne(self, user_batch,rating_batch,item_batch):
        user_e,item_e=self.model.computer()
        user_b = F.embedding(user_batch, user_e)
        item_b = F.embedding(item_batch, item_e)
        prediction = (user_b * item_b).sum(dim=-1)
        loss_part = self.mse_loss(prediction, rating_batch)
        l2_regulization = 0.1 * (user_b ** 2 + item_b ** 2).sum(dim=-1)#lastfm=0.01
        loss = loss_part + l2_regulization.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.cpu().item()

class FairnessScore:
    def __init__(self):
        item_feature=np.load('../../data/ml1m/data1t5/item_feature.npy',allow_pickle=True)
        user_feature=np.load('../../data/ml1m/data1t5/users_features_list.npy',allow_pickle=True)
        self.item_feature=item_feature
        self.user_feature=user_feature
    def get_Fairness_Score(self,userid,itemid,Attribute):
        if Attribute=='gender':
            item=self.item_feature[itemid,:2]
            user=self.user_feature[userid,:2]
            return np.sum(item*user)
        else:
            return

class FairAndPrivacy:
    def __init__(self,topk=5,usernum=6040,itemnum=3952):
        item_feature=np.load('../../data/ml1m/data1t5/item_feature.npy',allow_pickle=True)
        #user_feature=np.load('../../data/ml1m/data1t5/users_features_list.npy',allow_pickle=True)
        user_feature=np.load('../../data/ml1m/data1t5/users_features_3num.npy',allow_pickle=True)
        self.item_feature=item_feature
        #print(self.item_feature)
        # for iid in range(itemnum):
        #     if self.item_feature[iid,0]>0:
        #         self.item_feature[iid,0]=self.item_feature[iid,0]*0.3
        #     else:
        #         self.item_feature[iid, 0] = self.item_feature[iid, 0] * 0.7
        #print(self.item_feature)
        self.user_feature=user_feature
        self.topk=topk
        self.usernum=usernum
        self.itemnum=itemnum
    def get_Att_precision(self,user_embedding,item_embedding,Attribute):
        reclist=self.get_reclist(user_embedding,item_embedding)
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
                recitem = self.item_feature[rl,2:9]
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
        if Attribute=='occ':
            predict_Att = []
            for uid in range(self.usernum):
                rl = reclist[uid]
                recitem = self.item_feature[rl,9:]
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
            true_Att = self.user_feature[:, 2]
            predict_Att = np.array(predict_Att)
        #print(predict_Att)
        return precision_score(true_Att,predict_Att,average='micro'),recall_score(true_Att,predict_Att,average='micro')
    def get_reclist(self,user_embedding,item_embedding):
        item_embedding=item_embedding.T
        rating=user_embedding.dot(item_embedding)
        rating=torch.cuda.FloatTensor(rating)
        _,reclist=rating.topk(self.topk,dim=1)#reclist是(usernum,topk)
        return reclist

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 2048)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def UniformSample_original(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    user_num = dataset.traindataSize
    users = np.random.randint(0, dataset.n_user, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_item)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S), [total, sample_time1, sample_time2]

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n

    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset: BasicDataset
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')