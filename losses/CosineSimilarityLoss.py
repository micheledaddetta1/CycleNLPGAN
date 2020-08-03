import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from models import Transformer

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.loss = torch.nn.CosineEmbeddingLoss()

    def forward(self, sentence_feature_generated, sentence_feature_real):
        output = self.loss(sentence_feature_generated, sentence_feature_real)
        return output