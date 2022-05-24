import torch
import torch.nn as nn
"""
 @author Moses
 @discription  处理类别特征跟连续值特征，
"""
class SparseDenseFeature(nn.Module):

    def __init__(self, sparse_feature_dict, dense_feature_dict, emb_dim=128):
        """
        Args:
           sparse_feature_dict: user feature dict include: {feature_name: (feature_unique_num, feature_index)}
           dense_feature_dict: item feature dict include: {feature_name: (1, feature_index)}
        """
        super(SparseDenseFeature, self).__init__()
        self.sparse_feature_dict = sparse_feature_dict
        self.dense_feature_dict = dense_feature_dict

        for sparse_cate, num in self.sparse_feature_dict.items():
                # 一般sparse_feature_dict的unique值都是大于1的，所以 sparse_cate_feature_nums=len(sparse_feature_dict)
            setattr(self, sparse_cate, nn.Embedding(num[0], emb_dim))

    def forward(self, inputs):
        sparse_embed_list, dense_embed_list = list(), list()
        for sparse_feature, num in self.sparse_feature_dict.items():
            sparse_embed_list.append(getattr(self, sparse_feature)(inputs[:, num[1]].long()))

        for dense_feature, num in self.dense_feature_dict.items():
            dense_embed_list.append(inputs[:, num[1]].unsqueeze(1))

        # embedding 融合
        sparse_embed = torch.cat(sparse_embed_list, axis=1)
        dense_embed = torch.cat(dense_embed_list, axis=1)

        # hidden layer hidden size = len(sparse_feature_dict) * emb_dim + len(dense_feature_dict)
        h = torch.cat([sparse_embed, dense_embed], axis=1).float()
        return h


