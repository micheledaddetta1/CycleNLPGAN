import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):

    def __init__(self, in_dim, out_dim=2, n_layers=3):
        super(MultiLayerPerceptron, self).__init__()
        modules=[]
        self.in_dim=in_dim
        self.n_layers = n_layers

        in_d = in_dim
        out_d = int(in_dim/2)
        for n in range(n_layers):
            if n == n_layers-1:
                out_d = out_dim
            modules.append(nn.Linear(in_d, out_d, bias=True))
            modules.append(nn.ReLU())
            in_d = out_d
            out_d = int(in_d/2)
            if out_d < out_dim:
                out_d = out_dim
        self.layers = nn.Sequential(*modules)

    def forward(self, xb):
        x = xb.view(-1, self.in_dim)
        x = self.layers(x)
        return x

