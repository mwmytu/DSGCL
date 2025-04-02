import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType
import torch.nn.functional as F


class DSGCL(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DSGCL, self).__init__(config, dataset)
        self._user = dataset.inter_feat[dataset.uid_field]
        self._item = dataset.inter_feat[dataset.iid_field]

        # load parameters info
        self.embed_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.dropout = config["dropout"]
        self.temp = config["temp"]
        self.lambda_1 = config["lambda1"]
        self.lambda_2 = config["lambda2"]
        self.eps = config["eps"]
        self.l1 = config["l1"]
        self.l2 = config["l2"]
        self.k = config['k']
        self.act = nn.LeakyReLU(0.5)
        self.reg_loss = EmbLoss()

        # get the normalized adjust matrix
        self.adj_norm = self.coo2tensor(self.create_adjust_matrix())
        u, v = self.custom_matrix_factorization(self.adj_norm, self.k)
        self.mf_u = u
        self.mf_v = v

        torch.save(self.mf_u, 'user_mf.pt')
        torch.save(self.mf_v, 'item_mf.pt')

        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.embed_dim)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.embed_dim)))
        self.E_u_list = [None] * (self.n_layers + 1)
        self.E_i_list = [None] * (self.n_layers + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (self.n_layers + 1)
        self.Z_i_list = [None] * (self.n_layers + 1)
        self.G_u_list = [None] * (self.n_layers + 1)
        self.G_i_list = [None] * (self.n_layers + 1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0

        self.E_u = None
        self.E_i = None
        self.restore_user_e = None
        self.restore_item_e = None

        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def create_adjust_matrix(self):
        ratings = np.ones_like(self._user, dtype=np.float32)
        matrix = sp.csr_matrix(
            (ratings, (self._user, self._item)),
            shape=(self.n_users, self.n_items),
        ).tocoo()
        rowD = np.squeeze(np.array(matrix.sum(1)), axis=1)
        colD = np.squeeze(np.array(matrix.sum(0)), axis=0)
        for i in range(len(matrix.data)):
            matrix.data[i] = matrix.data[i] / pow(rowD[matrix.row[i]] * colD[matrix.col[i]], 0.5)
        return matrix

    def coo2tensor(self, matrix: sp.coo_matrix):
        indices = torch.from_numpy(
            np.vstack((matrix.row, matrix.col)).astype(np.int64))
        values = torch.from_numpy(matrix.data)
        shape = torch.Size(matrix.shape)
        x = torch.sparse.FloatTensor(indices, values, shape).coalesce().to(self.device)
        return x

    def sparse_dropout(self, matrix, dropout):
        if dropout == 0.0:
            return matrix
        indices = matrix.indices()
        values = F.dropout(matrix.values(), p=dropout)
        size = matrix.size()
        return torch.sparse.FloatTensor(indices, values, size)

    def custom_matrix_factorization(self, adj_norm, k, learning_rate=0.01, num_iterations=100):
        m, n = adj_norm.shape
        U = torch.randn(m, k, requires_grad=True, device=adj_norm.device)
        V = torch.randn(n, k, requires_grad=True, device=adj_norm.device)
        optimizer = torch.optim.Adam([U, V], lr=learning_rate)
        # 将 adj_norm 转换为 PyTorch 稠密张量
        adj_norm_dense = adj_norm.to_dense()

        for _ in range(num_iterations):
            optimizer.zero_grad()
            # 使用 U 和 V 计算预测评分
            pred_ratings = torch.mm(U, V.t())
            # 计算损失
            loss = torch.norm(adj_norm_dense - pred_ratings, 'fro')
            loss.backward()
            optimizer.step()
        # 返回左奇异矩阵 U 和右奇异矩阵 V 的张量形式
        return U.detach(), V.detach()

    def forward(self):
        for layer in range(1, self.n_layers + 1):
            # 混合噪声
            random_i_noise = self.l1 * torch.randn_like(self.E_i_list[layer - 1]).to(
                self.E_i_list[layer - 1].device) + self.l2 * torch.rand_like(self.E_i_list[layer - 1]).to(
                self.E_i_list[layer - 1].device)
            random_u_noise = self.l1 * torch.randn_like(self.E_u_list[layer - 1]).to(
                self.E_u_list[layer - 1].device) + self.l2 * torch.rand_like(self.E_u_list[layer - 1]).to(
                self.E_u_list[layer - 1].device)
            # GNN propagation
            self.Z_u_list[layer] = torch.spmm(self.adj_norm, self.E_i_list[layer - 1] + F.normalize(random_i_noise, dim=-1) * self.eps)
            self.Z_i_list[layer] = torch.spmm(self.adj_norm.transpose(0, 1), self.E_u_list[layer - 1] + F.normalize(random_u_noise, dim=-1) * self.eps)

            # self.Z_u_list[layer] = torch.spmm(self.sparse_dropout(self.adj_norm, self.dropout),
            #                                   self.E_i_list[layer - 1])
            # self.Z_i_list[layer] = torch.spmm(self.sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1),
            #                                   self.E_u_list[layer - 1])
            # aggregate
            self.E_u_list[layer] = self.Z_u_list[layer]
            self.E_i_list[layer] = self.Z_i_list[layer]

        # aggregate across wsdadwsd
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

        # torch.save(self.E_u, 'user_embeddings_with_cl.pt')
        # torch.save(self.E_i, 'item_embeddings_with_cl.pt')

        return self.E_u, self.E_i

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_list = interaction[self.USER_ID]
        pos_item_list = interaction[self.ITEM_ID]
        neg_item_list = interaction[self.NEG_ITEM_ID]
        E_u_norm, E_i_norm = self.forward()
        bpr_loss = self.calc_bpr_loss(E_u_norm, E_i_norm, user_list, pos_item_list, neg_item_list)
        ssl_loss = self.calc_ssl_loss(E_u_norm, E_i_norm, user_list, pos_item_list)
        total_loss = bpr_loss + ssl_loss
        return total_loss

    def calc_bpr_loss(self, E_u_norm, E_i_norm, user_list, pos_item_list, neg_item_list):
        u_e = E_u_norm[user_list]
        pi_e = E_i_norm[pos_item_list]
        ni_e = E_i_norm[neg_item_list]
        pos_scores = torch.mul(u_e, pi_e).sum(dim=1)
        neg_scores = torch.mul(u_e, ni_e).sum(dim=1)
        loss1 = -(pos_scores - neg_scores).sigmoid().log().mean()

        # reg loss
        loss_reg = 0
        for param in self.parameters():
            loss_reg += param.norm(2).square()
        loss_reg *= self.lambda_2
        return loss1 + loss_reg

    def calc_ssl_loss(self, E_u_norm, E_i_norm, user_list, pos_item_list):
        # calculate G_u_norm&G_i_norm
        for layer in range(1, self.n_layers + 1):
            # svd_adj propagation
            vt_ei = self.mf_v.transpose(0, 1) @ self.E_i_list[layer - 1]
            self.G_u_list[layer] = self.mf_u @ vt_ei
            ut_eu = self.mf_u.transpose(0, 1) @ self.E_u_list[layer - 1]
            self.G_i_list[layer] = self.mf_v @ ut_eu

        # aggregate across layer
        G_u_norm = sum(self.G_u_list)
        G_i_norm = sum(self.G_i_list)

        neg_score = torch.log(torch.exp(G_u_norm[user_list] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i_norm[pos_item_list] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((G_u_norm[user_list] * E_u_norm[user_list]).sum(1) / self.temp, -5.0, 5.0)).mean() + (
            torch.clamp((G_i_norm[pos_item_list] * E_i_norm[pos_item_list]).sum(1) / self.temp, -5.0, 5.0)).mean()

        ssl_loss = -pos_score + neg_score
        return self.lambda_1 * ssl_loss

    def predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        user = self.restore_user_e[interaction[self.USER_ID]]
        item = self.restore_item_e[interaction[self.ITEM_ID]]
        return torch.sum(user * item, dim=1)

    def full_sort_predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        user = self.restore_user_e[interaction[self.USER_ID]]
        return user.matmul(self.restore_item_e.T)
