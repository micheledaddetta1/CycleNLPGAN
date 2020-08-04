import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from models import Transformer


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.loss = torch.nn.CosineEmbeddingLoss()

    def forward(self, sentence_feature_generated, sentence_feature_real, target):
        output_list = list()
        output = 0
        for i, (real, generated) in enumerate(zip(sentence_feature_real, sentence_feature_generated)):
            output_list.append(self.loss(real.unsqueeze(0), generated.unsqueeze(0), target.unsqueeze(0)))
            output += output_list[i]
        dim = len(output_list)
        output /= dim

        return output
