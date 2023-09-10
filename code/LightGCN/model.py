import numpy as np

import world
import torch
from torch import nn

class LightGCN(nn.Module):
    def __init__(self,
                 config,
                 dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.ego_w=self.dataset.get_ego
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim*4, int(self.latent_dim), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            #nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(p=0.3),
            #nn.Linear(int(self.latent_dim*2), int(self.latent_dim*2), bias=True),
            #nn.LeakyReLU(0.2,inplace=True),
            # # nn.Dropout(p=0.3),
            # # nn.Sigmoid(),
            #nn.Linear(int(self.latent_dim*2), self.latent_dim, bias=True),
            # nn.Sigmoid(),
            # nn.LeakyReLU(0.2,inplace=True),
            # nn.Linear(self.out_dim, self.out_dim , bias=True)
            # nn.Sigmoid()
            )
        #####lastfm数据集需要解除上方注释

        if self.config['pretrain'] == 0:
            # nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            # nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            nn.init.normal_(self.embedding_user.weight, std=0.01)
            nn.init.normal_(self.embedding_item.weight, std=0.01)
            print('use xavier initilizer')
        else:
            # self.embedding_user.weight.data.copy_(torch.from_numpy(user_e))
            # self.embedding_item.weight.data.copy_(torch.from_numpy(item_e))
            print('use pretarined data')
        # usernotintrainset,item_not_in_trainset=np.load('../../code/code_lastfm/gcnModel/node_not_in_trainset.npy',allow_pickle=True)
        #
        # item_e=torch.FloatTensor(item_e)
        # user_e = torch.FloatTensor(user_e)
        # for itemid in item_not_in_trainset:
        #     self.embedding_item.weight.data[itemid]=item_e[itemid]
        # for userid in usernotintrainset:
        #     self.embedding_user.weight.data[userid] = user_e[userid]
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)#对所有节点进行聚合
            embs.append(all_emb)
        embedding=torch.cat((embs[0],embs[1],embs[2],embs[3]),-1)
        embedding=self.net(embedding)
        #print(embedding.shape)
        users, items = torch.split(embedding, [self.num_users, self.num_items])
        return users,items
        # embs = torch.stack(embs, dim=1)
        # # print(embs.size())
        # light_out = torch.mean(embs, dim=1)
        # users, items = torch.split(light_out, [self.num_users, self.num_items])
        # return users, items
    def getpretrainembedding(self):
        return self.embedding_user,self.embedding_item
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating


    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
