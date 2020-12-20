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
        output_list = list()
        output = 0

        for i, (real, generated) in enumerate(zip(sentence_feature_real, sentence_feature_generated)):
            output_list.append(((real - generated) ** 2.0).mean())
            output += output_list[i]
        dim = len(output_list)
        output /= dim
        return output

