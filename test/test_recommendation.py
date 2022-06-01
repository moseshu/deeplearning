import numpy as np
import torch
from recommendation.models.esmm import ESMM
from recommendation.models.mmoe import MMOE

def test_esmm():
    a = torch.from_numpy(np.array([[1, 2, 4, 2, 0.5, 0.1],
                                   [4, 5, 3, 8, 0.6, 0.43],
                                   [6, 3, 2, 9, 0.12, 0.32],
                                   [9, 1, 1, 1, 0.12, 0.45],
                                   [8, 3, 1, 4, 0.21, 0.67]]))
    sparse_cate_dict = {'user_id': (11, 0), 'user_list': (12, 3), 'user_num': (2, 4)}
    dense_cate_dict = {'item_id': (1, 1), 'item_cate': (1, 2), 'item_num': (1, 5)}
    esmm = ESMM(sparse_cate_dict, dense_cate_dict)
    model = ESMM(sparse_cate_dict, dense_cate_dict)
    p_ctr, p_ctcvr = model(a)
    assert p_ctr.shape == (5, 1)
    assert p_ctcvr.shape == (5, 1)

def test_mmoe():
    a = torch.from_numpy(np.array([[1, 2, 4, 2, 0.5, 0.1],
                                   [4, 5, 3, 8, 0.6, 0.43],
                                   [6, 3, 2, 9, 0.12, 0.32],
                                   [9, 1, 1, 1, 0.12, 0.45],
                                   [8, 3, 1, 4, 0.21, 0.67]]))
    sparse_cate_dict = {'user_id': (11, 0), 'user_list': (12, 3), 'user_num': (3, 4)}
    dense_cate_dict = {'item_id': (1, 1), 'item_cate': (1, 2), 'item_num': (1, 5)}
    model = MMOE(128, sparse_cate_dict, dense_cate_dict)


    p = model(a)

    print(p[1])
    print(p[0].shape)

if __name__ == '__main__':
    # test_esmm()

    test_mmoe()