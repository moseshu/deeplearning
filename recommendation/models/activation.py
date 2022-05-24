import torch.nn as nn
import torch

class Dice(nn.Module):
    """
    Input shape:
        - 2 dims: [batch_size, embedding_size(features)]
        - 3 dims: [batch_size, num_features, embedding_size(features)]
    Output shape:
        - 输出维度跟输入一样
    References
        - https://github.com/fanoping/DIN-pytorch,
    """

    def __init__(self, emb_size, dim=2, epsilon=1e-8, device='cpu'):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)).to(device))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        return out



class ActivationLayer(nn.Module):
    def __init__(self, act_name: str, hidden_size=None, dice_dim=2):
        super(ActivationLayer, self).__init__()
        self.act_name = act_name
        if act_name.lower() == 'sigmoid':
            self.act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            self.act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'dice':
            assert dice_dim
            self.act_layer = Dice(hidden_size, dice_dim)
        elif act_name.lower() == 'prelu':
            self.act_layer = nn.PReLU()

    def forward(self, x):
        return self.act_layer(x)
