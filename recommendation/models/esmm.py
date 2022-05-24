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
if __name__ == "__main__":
    import numpy as np
    a = torch.from_numpy(np.array([[1, 2, 4, 2, 0.5, 0.1],
                                   [4, 5, 3, 8, 0.6, 0.43],
                                   [6, 3, 2, 9, 0.12, 0.32],
                                   [9, 1, 1, 1, 0.12, 0.45],
                                   [8, 3, 1, 4, 0.21, 0.67]]))
    sparse_cate_dict = {'user_id': (11, 0), 'user_list': (12, 3), 'user_num': (2, 4)}
    dense_cate_dict = {'item_id': (1, 1), 'item_cate': (1, 2), 'item_num': (1, 5)}
    esmm = ESMM(sparse_cate_dict, dense_cate_dict)
    model = ESMM(sparse_cate_dict,dense_cate_dict)
    p_ctr, p_ctcvr = model(a)
    print(p_ctr)
    print(p_ctcvr)