import torch.nn as nn
from recommendation.activations.activation import ActivationLayer
from typing import List


class DNN(nn.Module):
    """
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - inputs_dim: input feature dim.
        - hidden_units:是一个list，例子：[128,64,32] 每个值都是每一层的units值.
        - activation: 目前只支持这几种激活函数 sigmoid,linear,relu,dice,prelu.
        - l2_reg:L2正则， float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - dropout_rate: float in [0,1). Fraction of the units to dropout.
        - use_bn: bool.是否使用BatchNorm1d
        - seed: A Python integer to use as random seed.
    References
        - https://github.com/shenweichen/DeepCTR/blob/master/deepctr/layers/core.py
    """

    def __init__(self, inputs_dim, hidden_units: List, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, use_activation=True, dice_dim=3, seed=1024):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.use_activation = use_activation
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + hidden_units

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [ActivationLayer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            x = self.linears[i](deep_input)

            if self.use_bn:
                x = self.bn[i](x)
            if self.use_activation:
                x = self.activation_layers[i](x)

            x = self.dropout(x)
            deep_input = x
        return deep_input
