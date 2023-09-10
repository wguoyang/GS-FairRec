import copy
import gc
import pandas as pd
import numpy as np
import time
import operator
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import inv
import pc_gender_train
user_num=6040#user_size
item_num=3952#item_size
GG=dict()
class RW:
    def __init__(self,X,Y,R):
        XX, YY = ['user_' + str(x) for x in X], ['item_' + str(y) for y in Y]
        # print(X)
        # print(Y)
        self.uimatrix=csr_matrix((R,(X,Y)),shape=(6041,3953))
        #print(self.uimatrix[1,661])3
        self.QG = self.get_graph(XX, YY)
        self.e_graph=[]



    def get_graph(self,X,Y):
        """
        Args:
            X: user id
            Y: item id
        Returns:
            graph:dic['user_id1':{'item_id1':1},  ... ]
        """
        item_user = dict()
        for i in range(len(X)):
            user = X[i]
            item = Y[i]
            if item not in item_user:
                item_user[item] = {}
            #item_user[item][user]=1
            item_user[item][user]=self.uimatrix[int(user[5:]),int(item[5:])]
        #print(item_user)

        user_item = dict()
        for i in range(len(Y)):
            user = X[i]
            item = Y[i]
            if user not in user_item:
                user_item[user] = {}
            #user_item[user][item]=1
            user_item[user][item]=self.uimatrix[int(user[5:]),int(item[5:])]
        #print(user_item)
        G = dict(item_user,**user_item)
        #print(G)
        return G

    # def get_ego_graph(self,uid):
    #
    #     G=copy.deepcopy(self.QG)
    #     user='user_'+str(uid)
    #     user_item=dict()
    #     user_item[user]=copy.deepcopy(G[user])
    #     item_user=dict()
    #     items=copy.deepcopy(G[user])
    #     begin= time.time()
    #     for it in items.keys():#遍历一阶邻居节点item_id
    #         if it not in item_user:
    #             item_user[it] =copy.deepcopy(G[it])
    #     for it in item_user.keys():
    #         for us in item_user[it].keys():
    #             if us not in user_item:
    #                 user_item[us]=copy.deepcopy(G[us]) #一阶邻居周围的用户加入自中心图
    #             for i in list(user_item[us].keys()):
    #                 if i not in item_user:
    #                     del user_item[us][i]
    #     # for us in user_item.keys():
    #     #     for it in user_item[us].keys():
    #     #         if it not in item_user:
    #     #             item_user[it]=G[it]#一阶邻居周围用户的周围物品加入自中心图
    #     EG=dict(item_user,**user_item)
    #     end=time.time()
    #     print('egotime',end-begin)
    #     # print(G['user_1'])
    #     return EG

    # def generate_ego_graph(self):
    #     for userID in range(user_num):
    #         print(userID)
    #         id=userID+1
    #         eg=self.get_ego_graph(id)
    #         self.e_graph.append(eg)
    #         gc.collect()
    #     np.save('ego_graph/user_ego_graph.npy',self.e_graph)
    # def RW_in_ego_graph(self, alpha, userID, max_depth):
    #     # rank = dict()
    #     G=self.get_ego_graph(userID)
    #     userID = 'user_' + str(userID)
    #     rank = {x: 0 for x in G.keys()}
    #     rank[userID] = 1
    #     # 开始迭代
    #     #begin = time.time()
    #     for k in range(max_depth):
    #         tmp = {x: 0 for x in G.keys()}
    #         # 取出节点i和他的出边尾节点集合ri
    #         for i, ri in G.items():
    #             # 取节点i的出边的尾节点j以及边E(i,j)的权重wij,边的权重都为1，归一化后就是1/len(ri)
    #             for j, wij in ri.items():
    #                 tmp[j] += alpha * rank[i] / (1.0 * len(ri))
    #         tmp[userID] += (1 - alpha)
    #         rank = tmp
    #     print(rank)
    #     # end = time.time()
    #     # print('use_time', end - begin)
    #     lst = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:10]#排序
    #     for ele in lst:
    #         print("%s:%.3f, \t" % (ele[0], ele[1]))

    def graph_to_m(self,G):
        """
        Returns:
            a coo_matrix sparse mat M
            a list,total user item points
            a dict,map all the point to row index
        """
        graph = G
        vertex = list(graph.keys())
        address_dict = {}
        total_len = len(vertex)
        for index in range(len(vertex)):
            address_dict[vertex[index]] = index
        row = []
        col = []
        data = []
        for element_i in graph:
            weight = round(1/len(graph[element_i]),3)####取下一个节点的概率
            sumi=0.0
            for k in graph[element_i].keys():
                sumi+=graph[element_i][k]
            row_index=  address_dict[element_i]
            for element_j in graph[element_i]:
                col_index = address_dict[element_j]
                row.append(row_index)
                col.append(col_index)
                #data.append(weight)
                data.append(round(graph[element_i][element_j]/sumi,4))#按评分的概率分布
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        m = coo_matrix((data,(row,col)),shape=(total_len,total_len))

        return m,vertex,address_dict


    def mat_all_point(self,m_mat,vertex,alpha):
        """
        get E-alpha*m_mat.T
        Args:
            m_mat
            vertex:total item and user points
            alpha:the prob for random walking
        Returns:
            a sparse
        """

        total_len = len(vertex)
        row = []
        col = []
        data = []
        for index in range(total_len):
            row.append(index)
            col.append(index)
            data.append(1)
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        eye_t = coo_matrix((data,(row,col)),shape=(total_len,total_len))
        #生成单位矩阵

        return eye_t.tocsr()-alpha*m_mat.tocsr().transpose()

    def RW_use_matrix(self, alpha, ID, isuser,K=10,use_matrix=True):
        """
        Args:
            alpha:the prob for random walking
            userID:the user to recom
            K:recom item num
        Returns:
            a dic,key:itemid ,value:pr score
        """
        G = self.QG

        m, vertex, address_dict = self.graph_to_m(G)
        #userID = 'item_' + str(ID)
        if isuser:
            stance='user_'+str(ID)
        else :
            stance='item_'+str(ID)

        #print('add',address_dict)
        if stance not in address_dict:
            return []
        score_dict = {}
        recom_dict = {}
        mat_all = self.mat_all_point(m,vertex,alpha)
        index = address_dict[stance]
        initial_list = [[0] for row in range(len(vertex))]
        initial_list[index] = [1]
        r_zero = np.array(initial_list)
        res = gmres(mat_all,r_zero,tol=1e-8)[0]
        for index in range(len(res)):
            point = vertex[index]
            if len(point.strip().split('_'))<2:
                continue
            # if point in G[userID]:
            #     continue
            score_dict[point] = res[index]
        #print(score_dict)
        # for it in list(score_dict.keys()):
        #     if score_dict[it] <va :#去掉噪声节点
        #         del score_dict[it]
        # for zuhe in sorted(score_dict.items(),key=operator.itemgetter(1),reverse=True)[:K]:
        #     point,score = zuhe[0],zuhe[1]
        #     recom_dict[point] = score
        # print(recom_dict)
        return score_dict

    def get_user_score(self,alpha,ID,isuser):
        score_dict=self.RW_use_matrix(alpha, ID,isuser)
        if isuser:
            stance='user_'+str(ID)
        else :
            stance='item_'+str(ID)
        one_order=self.QG[stance]
        print(len(one_order))
        for it in one_order.keys():
            print(it+str(score_dict[it]))


    # def aggregate(self,alpha, va,user_e,item_e):
    #     user_agg = np.zeros(shape=(user_num, 64), dtype=float)
    #     for userID in range(user_num) :
    #         print(userID)
    #         userid=userID+1
    #         score_dict=self.RW_use_matrix(alpha,userid,va)
    #         sum=0
    #         for it in score_dict.keys():
    #             sum=sum+score_dict[it]
    #             key=it
    #             if key[0]=='u':
    #                 s=key[5:]
    #                 id=int(s)-1
    #                 if id == userID:
    #                     sum=sum-score_dict[key]
    #                     continue
    #                 user_agg[userID]+=score_dict[key]*user_e[id]
    #             else:
    #                 s=key[5:]
    #                 id = int(s) - 1
    #                 user_agg[userID] += score_dict[key] * item_e[id]
    #         if userID==4168:
    #             user_agg[userID]=user_e[userID]
    #         else:
    #             user_agg[userID]=user_agg[userID]/sum
    #     np.save('ego_graph/user_ego_graph1.npy', user_agg)
    #     return user_agg


ratingsPath = '../../data/ml1m/ratings.dat'
ratingsDF = pd.read_csv(ratingsPath, index_col=None, sep='::', header=None,names=['user_id', 'movie_id', 'rating', 'timestamp'],engine='python')
X=ratingsDF['user_id']
#print(X)
R=ratingsDF['rating']
Y=ratingsDF['movie_id']
#print(Y)

#RW(X,Y).generate_ego_graph()
# users_emb_gcn = np.load('./gcnModel/user_emb_epoch79.npy',allow_pickle=True)
# #users_emb_gcn是一个矩阵6040行64列，也就是embedding向量是64维的
# items_emb_gcn = np.load('./gcnModel/item_emb_epoch79.npy',allow_pickle=True)

RW(X,Y,R).get_user_score(alpha=0.8,ID=1,isuser=True)
# user_e=RW(X,Y).aggregate(alpha=0.8,va=0.001, user_e=users_emb_gcn, item_e=items_emb_gcn)
# auc_one, auc_res = pc_gender_train.clf_gender_all_pre(-1, -1, user_e, 64)
# str_f1f1_res = '\t gender auc:' + str(round(np.mean(auc_res),4))+ '\t'
# print(str_f1f1_res)
