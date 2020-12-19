import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict


class MSELoss(nn.Module):
    """
    Computes the MSE loss between the computed sentence embedding and a target sentence embedding
    """
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, sentence_feature_real, sentence_feature_generated, target):

        output = ((sentence_feature_real - sentence_feature_generated) ** 2).mean()

        return output

