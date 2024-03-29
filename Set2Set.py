"""
Reference
Paper: https://arxiv.org/abs/1511.06391
Author's code: https://github.com/LisaAnne/set2set
PyG implementation: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/glob/set2set.html#Set2Set
"""
import torch
import dgl
from torch import tensor
from torch.nn.functional import softmax
class Set2Set(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 processing_steps, 
                 num_layers=1):
        super(Set2Set, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels,
                                  num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, bg, feat):
        batch_size = bg.batch_size
        x = bg.ndata[feat]
        batch = tensor([], dtype = torch.int64)
        batch_num_nodes = bg.batch_num_nodes
        for index, num in enumerate(batch_num_nodes):
            batch = torch.cat((batch, tensor(index).expand(num)))
        
        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = x.new_zeros(batch_size, self.out_channels)

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = torch.cat(list(map(lambda x: softmax(x, dim = 0),
                                   list(torch.split(e, batch_num_nodes)))),
                          dim = 0)
            bg.ndata['w'] = a
            r = dgl.sum_nodes(bg, feat, 'w')
            q_star = torch.cat([q, r], dim=-1)

        return q_star


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
