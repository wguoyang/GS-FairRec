import numpy as np
import utils
import torch.nn as nn
import torch
from scipy.sparse import csr_matrix
class TestPcd:
    def __init__(self):
        self.topks=[10]
        self.f = nn.Sigmoid()
        test_file='LightGCN_train_test_set/test.txt'
        train_file='LightGCN_train_test_set/train.txt'
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.n_user = 0  # 用户数量
        self.m_item = 0  # 物品数量
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')  # 按空格拆分
                    items = [int(i) for i in l[1:]]  # 第一列是用户id，后面是与该用户产生过交互的物品
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))  # x:uid
                    trainItem.extend(items)  # y:item
                    # (x,y)确定一组交互行为
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        self.testDict=self.__build_test()
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))  # 创建一个稀疏矩阵
        self.allPos = self.getUserPosItems(list(range(self.n_user)))
    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):#enumerate能够在遍历时返回索引，即i是索引，item是对象
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])#取到每个用户交互物品的下标，[1]的作用是？
        return posItems
    def getUsersRating(self, user_e,item_e,users):
        all_users=user_e
        all_items = item_e
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    def test_one_batch(self,X):
        topks=self.topks
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r = utils.getLabel(groundTrue, sorted_items)
        pre, recall, ndcg = [], [], []
        for k in topks:
            ret = utils.RecallPrecision_ATk(groundTrue, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
        return {'recall':np.array(recall),
                'precision':np.array(pre),
                'ndcg':np.array(ndcg)}
    def Test(self, user_e,item_e,multicore=0):
        u_batch_size = 100
        testDict= self.testDict
        # eval mode with no dropout
        #Recmodel = Recmodel.eval()
        topks=self.topks
        max_K = max(topks)
        results = {'precision': np.zeros(len(topks)),
                   'recall': np.zeros(len(topks)),
                   'ndcg': np.zeros(len(topks))}
        with torch.no_grad():
            users = list(testDict.keys())
            test_items = list(set(self.testItem)) #set the testitem list only in test dataset
            allitem_list = list(range(self.m_item))
            outitems = list(set(allitem_list) - set(test_items))
            auc_record = []
            # ratings = []
            total_batch = len(users) // u_batch_size + 1
            allPos = self.getUserPosItems(users)
            groundTrue = [testDict[u] for u in users]
            users_gpu = torch.Tensor(users).long()
            users_gpu = users_gpu.to('cuda')

            rating = self.getUsersRating(user_e,item_e,users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
                exclude_index.extend([range_i] * len(outitems))
                exclude_items.extend(outitems)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            x = [rating_K.cpu(), groundTrue]
            result = self.test_one_batch(x)

            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            #results['auc'] = np.mean(auc_record)
            #rmse
            # user_embed, item_embed = Recmodel.computer()
            # user_embed = user_embed.cpu().detach().numpy()
            # item_embed = item_embed.cpu().detach().numpy()
            # test_rating_dict=np.load('../data/ml1m/testing_ratings_dict.npy',allow_pickle=True)
            # pre_all = []
            # label_all = []
            # for pair_i in test_rating_dict:
            #     u_id, r_v, i_id = pair_i
            #     pre_get = np.sum(user_embed[u_id] * item_embed[i_id])
            #     pre_all.append(pre_get)
            #     label_all.append(r_v)
            # r_test = rmse(np.array(pre_all), np.array(label_all))
            # res_test = round(np.mean(r_test), 4)
            # results['RMSE']=res_test
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