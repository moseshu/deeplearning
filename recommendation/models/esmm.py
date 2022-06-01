import torch
import torch.nn as nn
from recommendation.featureprocess.sparse_dense_feature import SparseDenseFeature
class CtrNetwork(nn.Module):
    """NN for CTR prediction"""

    def __init__(self, input_dim, out_dim=128, drop=0.5):
        super(CtrNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=out_dim),
            nn.ReLU(),
            nn.Linear(in_features=out_dim, out_features=1),
            nn.Dropout(drop)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        p = self.mlp(inputs)
        return self.sigmoid(p)


class CvrNetwork(nn.Module):
    """NN for CVR prediction"""

    def __init__(self, input_dim, out_dim=128, drop=0.5):
        super(CvrNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=out_dim),
            nn.ReLU(),
            nn.Linear(in_features=out_dim, out_features=1),
            nn.Dropout(drop)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        p = self.mlp(inputs)
        return self.sigmoid(p)


class ESMM(nn.Module):
    """ESMM
        例子：如果dense_feature为0，代表只有大类特征，无需合并embedding跟连续值特征
    """

    def __init__(self, sparse_feature_dict, dense_feature_dict, out_dim=128, emb_dim=128, drop=0.5):
        super(ESMM, self).__init__()
        self.sparse_dense_feature = SparseDenseFeature(sparse_feature_dict, dense_feature_dict, emb_dim)

        input_dim = len(sparse_feature_dict) * emb_dim + len(dense_feature_dict)

        self.ctr = CtrNetwork(input_dim, out_dim, drop)
        self.cvr = CvrNetwork(input_dim, out_dim, drop)

    def forward(self, inputs):
        # embedding
        h = self.sparse_dense_feature(inputs)
        # Predict pCTR

        p_ctr = self.ctr(h)

        # Predict pCVR
        p_cvr = self.cvr(h)

        # Predict pCTCVR
        p_ctcvr = torch.mul(p_ctr, p_cvr)
        return p_ctr, p_ctcvr
