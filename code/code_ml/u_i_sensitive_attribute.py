import torch
import numpy as np
from scipy.sparse import csr_matrix
###ml1m####
fp=0.24
selectp=0.5
train_rating,count=np.load('../../data/ml1m/data1t5/training_ratings_dict.npy',allow_pickle=True)#[uid,rat,iid]id从0开始
user_features=np.load('../../data/ml1m/data1t5/users_features_3num.npy',allow_pickle=True)#6040*3
u_i_dict,i_u_dict={},{}
for uid in range(6040):
    u_i_dict[uid]=[]
for iid in range(3952):
    i_u_dict[iid]=[]
for idict in train_rating.keys():
    [userid,rat,itemid]=train_rating[idict]
    u_i_dict[userid].append(itemid)
    i_u_dict[itemid].append(userid)

user_adj_score=np.load('./RW_adj_score/FGuser_adj.npy',allow_pickle=True).item()
item_adj_score=np.load('./RW_adj_score/FGitem_adj.npy',allow_pickle=True).item()
user_un_f_id={}#未过滤的节点
for uid in range(6040):
    us=user_adj_score['user_'+str(uid)]
    f_num=int(len(us)*fp)
    s_num=int(len(us)*selectp)
    if f_num<1:
        f_num=1
    #user_f_num.append(f_num)
    #dit = sorted(us.items(), key=lambda x: x[1])[:f_num]
    dit = sorted(us.items(), key=lambda x: x[1])[f_num:]
    lin_id=[]
    for it,sc in dit:
        lin_id.append(int(it[5:]))
    user_un_f_id[uid]=lin_id
item_un_f_id={}
for iid in range(3952):
    iscor=item_adj_score['item_'+str(iid)]
    f_num = int(len(iscor) * fp)
    if f_num < 1:
        f_num = 1
    dit = sorted(iscor.items(), key=lambda x: x[1])[f_num:]
    lin_id = []
    for it, sc in dit:
        lin_id.append(int(it[5:]))
    item_un_f_id[iid] = lin_id
"""

"""
# user_un_interest_id={}#不感兴趣的物品集合
# for uid in range(6040):
#     us=user_adj_score['user_'+str(uid)]
#     f_num=int(len(us)*fp)
#     s_num=int(len(us)*selectp)
#     if f_num<1:
#         f_num=1
#     #user_f_num.append(f_num)
#     #dit = sorted(us.items(), key=lambda x: x[1])[:f_num]
#     dit = sorted(us.items(), key=lambda x: x[1])[f_num:f_num+s_num]
#     lin_id=[]
#     for it,sc in dit:
#         lin_id.append(int(it[5:]))
#     user_un_interest_id[uid]=lin_id
# user_adj_att_sim={}#相似属性的物品集合
sen=2#0,1,2
# for uid in range(6040):
#     us = user_adj_score['user_' + str(uid)]
#     s_num = int(len(us) * selectp)
#     usen=user_features[uid,sen]
#     item_sim={}
#     for iid in user_un_f_id[uid]:#遍历一阶邻居
#         sumitem = 0
#         issen=0
#         for nebuser in i_u_dict[iid]:
#             sumitem+=1
#             if user_features[nebuser,sen]==usen:
#                 issen+=1
#         percent=issen/sumitem
#         item_sim[iid]=percent
#     dit = sorted(item_sim.items(), key=lambda x: x[1],reverse=True)[:s_num]#从大到小排
#     lin_id = []
#     for it, sc in dit:
#         lin_id.append(it)
#     user_adj_att_sim[uid]=lin_id
# user_select={}
# for uid in range(6040):
#     user_select[uid]=list(set(user_adj_att_sim[uid]) & set(user_un_interest_id[uid]))
# np.save('./select/user_select_0.5_occ.npy',user_select)

#secend_order=np.zeros(shape=(6040,),dtype=int)
sumz=0.
sumb=0.
for uid in range(6040):
    usen=user_features[uid,sen]
    sumsec=0
    sumsensame=0
    # secend_order = np.zeros(shape=(6040,), dtype=int)
    # for iid in u_i_dict[uid]:
    #     for sec_uid in i_u_dict[iid]:
    #         secend_order[sec_uid]=1
    # for sec_uid in range(6040):
    #     if secend_order[sec_uid]==1:
    #         sumsec+=1
    #         if user_features[sec_uid,sen]==usen:
    #             sumsensame+=1
    # sumz=sumz+sumsensame/sumsec
    sumsec = 0
    sumsensame = 0
    secend_order = np.zeros(shape=(6040,), dtype=int)
    for iid in user_un_f_id[uid]:
        for sec_uid in item_un_f_id[iid]:
            secend_order[sec_uid] = 1
    for sec_uid in range(6040):
        if secend_order[sec_uid] == 1:
            sumsec += 1
            if user_features[sec_uid, sen] == usen:
                sumsensame += 1
    sumb = sumb + sumsensame / sumsec
print(sumz/6040)
print(sumb/6040)
# sumw=0.0
# for uid in range(6040):
#     usen=user_features[uid,sen]
#     sumn=0.
#     for iid in user_un_f_id[uid]:
#         sumitem=0
#         issen=0
#         for nebuser in i_u_dict[iid]:
#             sumitem+=1
#             if user_features[nebuser,sen]==usen:
#                 issen+=1
#         percent=issen/sumitem
#         sumn=sumn+percent
#     sumw=sumw+sumn/len(user_un_f_id[uid])
# print(sumw/6040)


# for i,iid in enumerate(u_i_dict[5]):
#     sum=0
#     genderu=0
#     for j,uid in enumerate(i_u_dict[iid]):
#         sum+=1
#         if user_features[uid,0]==gen:
#             genderu+=1
#     print(genderu/sum)
#     f_0_id=user_f_id[5]
#     k=0
#     for index,da in enumerate(f_0_id):
#         if iid==da:
#             k=1
#             break
#     if k==0 :
#         print('否')
#     else :
#         print("是")


