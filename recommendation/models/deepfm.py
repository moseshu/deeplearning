import torch.nn as nn
import torch
import numpy as np
class DeepFM(nn.Module):
    def __init__(self, cate_fea_nuniqs, nume_fea_size=0, emb_size=8,
                 hid_dims=[256, 128], num_classes=1, dropout=[0.2, 0.2]):
        """
        cate_fea_nuniqs: 类别特征的唯一值个数列表，也就是每个类别特征的vocab_size所组成的列表
        nume_fea_size: 数值特征的个数，该模型会考虑到输入全为类别型，即没有数值特征的情况
        """
        super().__init__()
        self.cate_fea_size = len(cate_fea_nuniqs)
        self.nume_fea_size = nume_fea_size

        """FFM部分"""
        # 一阶
        if self.nume_fea_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.nume_fea_size, 1)  # 数值特征的一阶表示

        self.fm_1st_order_sparse_emb = torch.nn.Embedding(sum(cate_fea_nuniqs), 1)  # [40,1]
        self.bias = torch.nn.Parameter(torch.zeros((1,)))
        self.offsets = np.array((0, *np.cumsum(cate_fea_nuniqs)[:-1]), dtype=np.compat.long)  # 类别特征的一阶表示

        # 二阶
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(cate_fea_nuniqs), emb_size) for _ in range(self.cate_fea_size)  # 类别特征的二阶表示
        ])
        self.offsets = np.array((0, *np.cumsum(cate_fea_nuniqs)[:-1]), dtype=np.compat.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

        """DNN部分"""
        self.all_dims = [self.cate_fea_size*self.cate_fea_size * emb_size] + hid_dims
        self.dense_linear = nn.Linear(self.nume_fea_size,
                                      self.cate_fea_size * self.cate_fea_size * emb_size)  # 数值特征的维度变换到FM输出维度一致
        self.relu = nn.ReLU()
        # for DNN
        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_' + str(i), nn.Linear(self.all_dims[i - 1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i - 1]))
        # for output
        self.dnn_linear = nn.Linear(hid_dims[-1], num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse, X_dense=None):
        """           【2048 26】    【 2048 13】
        X_sparse: 类别型特征输入  [bs, cate_fea_size]
        X_dense: 数值型特征输入（可能没有）  [bs, dense_fea_size]
        """
        X_sparse = X_sparse + X_sparse.new_tensor(self.offsets, dtype=np.compat.long).unsqueeze(0)

        """FM 一阶部分"""
        fm_1st_sparse_emb = self.fm_1st_order_sparse_emb(X_sparse)  # [2048 26 1]
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_emb, dim=1) + self.bias  # [2048 1]

        if X_dense is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(X_dense)  # [2048 1]
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res  # [2048 1]
        else:
            fm_1st_part = fm_1st_sparse_res  # [2048, 1]

        """FM 二阶部分"""
        xs = [self.embeddings[i](X_sparse) for i in range(self.cate_fea_size)]  # [26, 2048 ,26 ,8]
        ix = list()
        for i in range(self.cate_fea_size - 1):
            for j in range(i + 1, self.cate_fea_size):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)  # [325 2048 8]
        fm_2nd_part = torch.sum(torch.sum(ix, dim=1), dim=1, keepdim=True)  # [2048 1]
        """DNN部分"""
        fm_2nd_concat_1d = torch.cat(xs, dim=1)  # [2048, 26*26, 8]
        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)  # [2048, 26*26*8]

        if X_dense is not None:
            dense_out = self.relu(self.dense_linear(X_dense))  # [2048, 26*26*8]
            dnn_out = dnn_out + dense_out  # [2048, 26*26*8]

        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)

        dnn_out = self.dnn_linear(dnn_out)  # [2048, 1]
        out = fm_1st_part + fm_2nd_part + dnn_out  # [2048, 1]
        out = self.sigmoid(out)
        return out