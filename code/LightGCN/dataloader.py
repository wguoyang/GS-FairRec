
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time
import copy

class Loader():
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, config=world.config, path="../../data/lastfm"):
        # train or test
        cprint(f'loading [{path}]')
        self.ifRW = config['ifRW']
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0#用户数量
        self.m_item = 0#物品数量
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        trainrating,testrating=[],[]
        self.traindataSize = 0
        self.testDataSize = 0
        LightGCN=False
        if LightGCN==True:
            with open(train_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')#按空格拆分
                        items = [int(i) for i in l[1:]]#第一列是用户id，后面是与该用户产生过交互的物品
                        uid = int(l[0])
                        trainUniqueUsers.append(uid)
                        trainUser.extend([uid] * len(items))#x:uid
                        trainItem.extend(items)#y:item
                        #(x,y)确定一组交互行为
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
        else:
            train_rating_dict,train_count=np.load('../../data/ml1m/data1t5/training_ratings_dict.npy',allow_pickle=True)
            test_rating_dict,test_count=np.load('../../data/ml1m/data1t5/testing_ratings_dict.npy',allow_pickle=True)
            # train_rating_dict, train_count = np.load('../../data/lastfm/data1t5/training_ratings_dict.npy',
            #                                          allow_pickle=True)
            # test_rating_dict, test_count = np.load('../../data/lastfm/data1t5/testing_ratings_dict.npy',
            #                                        allow_pickle=True)
            #换数据集需要调整
            for i in train_rating_dict.keys():
                [userid,rat,itemid]=train_rating_dict[i]
                #rat=rat+1
                trainUser.append(userid)
                trainItem.append(itemid)
                trainrating.append(rat)
                self.m_item=max(self.m_item, itemid)
                self.n_user = max(self.n_user, userid)
            self.traindataSize = train_count
            self.trainUser = np.array(trainUser)
            self.trainItem = np.array(trainItem)
            self.trainrating=np.array(trainrating)


            for i in test_rating_dict.keys():
                [userid,rat,itemid]=test_rating_dict[i]
                #rat=rat+1
                testUser.append(userid)
                testItem.append(itemid)
                testrating.append(rat)
                self.m_item=max(self.m_item, itemid)
                self.n_user = max(self.n_user, userid)
            self.testDataSize = test_count
            self.testUser = np.array(testUser)
            self.testItem = np.array(testItem)
            self.testrating = np.array(testrating)

            self.m_item+=1
            self.n_user+=1

        # for uid in set(self.testUser):
        #     if uid not in self.trainUser:
        #         print(uid)
        # for iid in set(self.testItem):
        #     if iid not in self.trainItem:
        #         print(iid)


        if self.ifRW:
            print('use Node Filtering')
            self.user_f_num, self.user_f_id, self.item_f_num, self.item_f_id = self.RW_filter(userp=config['userfp'],
                                                                                          itemp=config['itemfp'],method='MF')
        else :
            print('not use Node Filtering')
        self.Graph = None
        print(f"{self.traindataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.traindataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))#创建一个稀疏矩阵
        # self.UserItemNet = csr_matrix((self.trainrating, (self.trainUser, self.trainItem)),
        #                               shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()#横向求和，去维度
        #self.users_D=self.users_D-self.user_f_num#随机游走过滤噪声节点
        self.users_D[self.users_D == 0.] = 1.#防止没有产生任何交互行为的用户求和后为0，导致分母为0
        self.users_D=np.power(self.users_D,-1)
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        #self.items_D=self.items_D-self.item_f_num#随机游走过滤噪声节点
        self.items_D[self.items_D == 0.] = 1.
        self.items_D = np.power(self.items_D, -1)
        self.users_D=self.users_D.reshape((self.n_user,1))
        self.users_D=torch.cuda.FloatTensor(self.users_D)
        self.items_D=self.items_D.reshape((self.m_item,1))
        self.items_D=torch.cuda.FloatTensor(self.items_D)
        self.ego=torch.cat((self.users_D,self.items_D),0)

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def get_ego(self):
        return self.ego
    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            #如果有已有文件直接加载，如果没有执行代码后保存数据
            # try:
            #     pre_adj_mat = sp.load_npz('../../data/lastfm/adj_mat'+'/s_pre_adj_mat_RW.npz')
            #     print("successfully loaded...")
            #     norm_adj = pre_adj_mat
            # except:
            print("generating adjacency matrix")
            s = time()
            #adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            # adj_mat = adj_mat.tolil()
            #
            # R = self.UserItemNet.tolil()#矩阵拼接先转为lil型，对矩阵中的值进行大量访问时先转为lil型
            #
            # adj_mat[:self.n_users, self.n_users:] = R
            # adj_mat[self.n_users:, :self.n_users] = R.T
            R=self.UserItemNet.tocoo()
            Z1=sp.coo_matrix((self.n_user,self.n_user),dtype=int)
            Z2=sp.coo_matrix((self.m_item,self.m_item),dtype=int)
            adj_mat=sp.bmat([[Z1,R],[R.T,Z2]])

            """
              0  R
            R.T  0
            """
            adj_mat = adj_mat.todok()
            # ===
            if self.ifRW:
                for uid in self.user_f_id.keys():
                    for iid in self.user_f_id[uid]:
                        adj_mat[uid,self.n_users+iid]=0
                for iid in self.item_f_id.keys():
                    for uid in self.item_f_id[iid]:
                        adj_mat[iid+self.n_users,uid]=0
            # ===
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()#开根号
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)#转为对角矩阵

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()#csr,稀疏矩阵的一种存储格式，最节省空间
            end = time()
            print(f"costing {end - s}s, saved norm_mat...")
            #sp.save_npz('../../data/lastfm/adj_mat'+ '/s_pre_adj_mat_RW.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)#转为tensor型的稀疏矩阵
                self.Graph = self.Graph.coalesce().to(world.device)#转到GPU上
                print("don't split the matrix")
        return self.Graph

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

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])#取到每个用户交互物品的下标，[1]的作用是？
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
    def RW_filter(self,userp,itemp,method):
        #参数，用户邻居节点过滤的比例，物品邻居节点过滤的比例
        #返回过滤掉的节点个数，以及过滤节点的id
        IFRW = False
        if method=='MF':
            print('MF')
            user_adj_score=np.load('../code_ml/RW_adj_score/USERCF_FGuser_adj.npy',allow_pickle=True).item()
            item_adj_score=np.load('../code_ml/RW_adj_score/USERCF_FGitem_adj.npy',allow_pickle=True).item()
        elif method=='RW':
            IFRW = True
            print('RW')
            user_adj_score = np.load('../code_ml/RW_adj_score/FGuser_adj.npy', allow_pickle=True).item()
            item_adj_score = np.load('../code_ml/RW_adj_score/FGitem_adj.npy', allow_pickle=True).item()
        else:
            print('Path')
            user_adj_score = np.load('../code_ml/RW_adj_score/NoP_FGuser_adj.npy', allow_pickle=True).item()
            item_adj_score = np.load('../code_ml/RW_adj_score/NoP_FGitem_adj.npy', allow_pickle=True).item()

        user_f_num=[]
        user_f_id={}
        for uid in range(self.n_user):
            if IFRW:
                us=user_adj_score['user_'+str(uid)]
            else:
                us=user_adj_score[uid]
            f_num=int(len(us)*userp)
            if f_num<1:
                f_num=0
            user_f_num.append(f_num)
            dit = sorted(us.items(), key=lambda x: x[1])[:f_num]
            lin_id=[]
            for it,sc in dit:
                if IFRW:
                    lin_id.append(int(it[5:]))
                else:
                    lin_id.append(int(it))
            user_f_id[uid]=lin_id

        item_f_num=[]
        item_f_id={}
        for iid in range(self.m_item):
            if IFRW:
                iscor=item_adj_score['item_'+str(iid)]
            else:
                iscor=item_adj_score[iid]
            if len(iscor)==0:
                item_f_num.append(0)
                item_f_id[iid]=[]
                continue
            f_num=int(len(iscor)*itemp)
            if f_num < 1:
                f_num = 0
            item_f_num.append(f_num)
            dit=sorted(iscor.items(), key=lambda x: x[1])[:f_num]
            lin_id=[]
            for it,sc in dit:
                if IFRW:
                    lin_id.append(int(it[5:]))
                else:
                    lin_id.append(int(it))
            item_f_id[iid] = lin_id
        return user_f_num,user_f_id,item_f_num,item_f_id