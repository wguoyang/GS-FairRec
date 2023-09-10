import copy
import gc
import pandas as pd
import numpy as np
import time
import operator
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import gmres
# path="../data/ml1m"
# train_file = path + '/train.txt'
# test_file = path + '/test.txt'
# ratingsPath = '../data/ml1m/ratings.dat'
# ratingsDF = pd.read_csv(ratingsPath, index_col=None, sep='::', header=None,names=['user_id', 'movie_id', 'rating', 'timestamp'],engine='python')
# X=ratingsDF['user_id']
# #print(X)
# R=ratingsDF['rating']
# Y=ratingsDF['movie_id']
# uimatrix=csr_matrix((R,(X,Y)),shape=(6041,3953))
# print(uimatrix[2665,3554])
# print(uimatrix[1,661])
# testUser=[]
# testItem=[]
# testdataSize=0
# m_item=0
# n_user=0
# testdataSize=0
# with open(test_file) as f:
#     for l in f.readlines():
#         if len(l) > 0:
#             l = l.strip('\n').split(' ')
#             items = [int(i) for i in l[1:]]
#             uid = int(l[0])
#             testUser.extend([uid] * len(items))
#             testItem.extend(items)
#             m_item = max(m_item, max(items))
#             n_user = max(n_user, uid)
#             testdataSize += len(items)
#
# testUser = np.array(testUser)
# testItem = np.array(testItem)
# test_rating_array=[]
# for i,ele in enumerate(testUser):
#     rating=uimatrix[ele+1,testItem[i]+1]
#     piece=[ele,rating,testItem[i]]
#     test_rating_array.append(piece)
#
# np.save(path+'/testing_ratings_dict.npy',test_rating_array)

######sushi dataset process#######
# file='../data/sushi3/sushi3b.5000.10.score'
# train_rating_dict={}
# test_rating_dict={}
# train_count=0
# test_count=0
# uid=0
# with open(file) as f:
#     for l in f.readlines():
#         l = l.strip('\n').split(' ')  # 按空格拆分
#         lint = [int(i) for i in l]  # 第一列是用户id，后面是与该用户产生过交互的物品
#         lz=[]
#         count=0
#         for iid,rat in enumerate(lint):
#             if rat>-1:
#                 if count<8:
#                     train_rating_dict[train_count]=[uid,rat+1,iid]
#                     train_count+=1
#                 else:
#                     test_rating_dict[test_count]=[uid,rat+1,iid]
#                     test_count+=1
#                 count+=1
#         uid+=1
# np.save('../data/sushi3/train_rating_dict.npy',[train_rating_dict,train_count])
# np.save('../data/sushi3/test_rating_dict.npy',[test_rating_dict,test_count])

# file='../data/sushi3/sushi3.udata'
# user_feature=[]
# with open(file) as f:
#     for l in f.readlines():
#         l = l.strip('\n').split('\t')  # 按空格拆分
#         lint = [int(i) for i in l]  # 第一列是用户id，后面是与该用户产生过交互的物品
#         user_feature.append([lint[1],lint[2]])
# np.save('../data/sushi3/user_feature.npy',user_feature)

# train_rating_dict, train_count = np.load('../data/sushi3/train_rating_dict.npy',
#                                                      allow_pickle=True)
# test_rating_dict, test_count = np.load('../data/sushi3/test_rating_dict.npy',
#                                                    allow_pickle=True)

# ls=[80,81,82,83,86,87,89,90,91,93,94,95,96,97,98,99]
# deleteindex=[]
# for index in range(10000):
#     if test_rating_dict[index][2] in ls:
#         train_rating_dict[train_count]=test_rating_dict[index]
#         train_count+=1
#         deleteindex.append(index)
# test_set={}
# count=0
# for index in range(10000):
#     if index not in deleteindex:
#         test_set[count]=test_rating_dict[index]
#         count+=1
#
# np.save('../data/sushi3/train_rating_dict_41191_.npy',[train_rating_dict,train_count])
# np.save('../data/sushi3/test_rating_dict_8809_.npy',[test_set,count])

# user_feature=np.load('../data/sushi3/user_feature.npy',
#                                                    allow_pickle=True)
# print()
# rw_user=np.load('../data/sushi3/RW_adj_score_user.npy',
#                                                    allow_pickle=True)
# rw_item=np.load('../data/sushi3/RW_adj_score_item.npy',allow_pickle=True)
# print()
