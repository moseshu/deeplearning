import torch.nn as nn
import torch
from recommendation.featureprocess.sparse_dense_feature import SparseDenseFeature
from recommendation.models.dnn import DNN

"""
 @author Moses
 @discription 
"""


class MMOE(nn.Module):

    def __init__(self, input_dim, sparse_feature_dict, dense_feature_dict,
                 num_experts=3,
                 num_tasks=2,
                 use_bias=False,
                 expert_hidden_units=[256, 128],
                 tower_hidden_units=[64],
                 gate_hidden_units=[128],
                 l2_reg=0,
                 seed=1024,
                 dropout=0,
                 activation='sigmoid',
                 use_bn=True, ):
        """

        :param input_dim: 类别特征embedding size
        :param sparse_feature_dict: {feature_name: (feature_unique_num, feature_index)}
        :param dense_feature_dict: {feature_name: (1, feature_index)}
        :param num_experts:
        :param number_tasks: (cvr, ctcvr)
        :param expert_hidden_units: list ,dnn 网络的units
        :param tower_hidden_units: list, dnn 网络的units
        :param gate_hidden_units: list , dnn 网络的units
        :param l2_reg_embedding:
        :param l2_reg:
        :param seed:
        :param dropout:
        :param activation:目前只有sigmoid,relu,prelu,linear,dice
        :param use_bn:
        References
            - https://github.com/shenweichen/DeepCTR/blob/master/deepctr/models/multitask/mmoe.py
        """
        super(MMOE, self).__init__()
        if num_experts <= 1:
            raise ValueError("num_experts 的值必须大于 1")
        self.sparse_dense_feature = SparseDenseFeature(sparse_feature_dict, dense_feature_dict, input_dim)
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.expert_hidden_units = expert_hidden_units
        self.tower_hidden_units = tower_hidden_units
        self.gate_hidden_units = gate_hidden_units
        self.l2_reg = l2_reg
        self.seed = seed
        self.dropout = dropout
        self.activation = activation
        self.use_bn = use_bn
        self.use_bias = use_bias
        hidden_size = input_dim * len(sparse_feature_dict) + len(dense_feature_dict)
        self.experts_list = nn.ModuleList([
            DNN(hidden_size, expert_hidden_units, activation, l2_reg, dropout, use_bn) for i in range(num_experts)
        ])
        self.gate_dnn = DNN(hidden_size, gate_hidden_units, activation, l2_reg, dropout, use_bn)
        self.gate_bias = [nn.Parameter(torch.rand(num_experts),requires_grad=True) for _ in range(num_tasks)]

    def forward(self, x: torch.Tensor):
        dnn_input_hidden = self.sparse_dense_feature(x)

        experts_out = []
        # expert layer
        for i, dnn_layer in enumerate(self.experts_list):
            x = dnn_layer(dnn_input_hidden)
            experts_out.append(x)
        experts_concat = torch.stack(experts_out, axis=1)  # batch,num_experts,dim

        # mmoe outputs
        mmoe_outs = []
        number_tasks = self.num_tasks
        for i in range(number_tasks):
            gate_input = self.gate_dnn(dnn_input_hidden)
            input_size = gate_input.shape[-1]

            gate_out = nn.Linear(input_size, self.num_experts)(gate_input)
            if self.use_bias:
                gate_out += self.gate_bias[i].to(x.device)
            gate_out = nn.Softmax(dim=-1)(gate_out) # batch, num_experts

            gate_out = torch.unsqueeze(gate_out, dim=-1) # batch, num_experts, 1
            gate_mul_expert = torch.mul(experts_concat, gate_out)

            # print("gate_mul_expert size=", gate_mul_expert.shape) #batch,num_experts,dim
            gate_sum = torch.sum(gate_mul_expert, dim=1)

            mmoe_outs.append(gate_sum)

        task_outs = []  # p_ctr = task_outs[0] ,p_ctcvr = task_outs[1]
        for i in range(number_tasks):
            input_size = mmoe_outs[i].shape[-1]
            tower_dnn = DNN(input_size, self.tower_hidden_units, self.activation, self.l2_reg, self.dropout, self.use_bn)
            tower_output = tower_dnn(mmoe_outs[i])
            # print("tower_output=",tower_output.shape)
            logit = nn.Linear(tower_output.shape[-1], 1)(tower_output)

            # print("logit size", logit.shape)
            task_outs.append(logit)
        return task_outs


