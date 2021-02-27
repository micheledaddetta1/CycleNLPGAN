import torch
from torch import nn


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.loss = torch.nn.CosineEmbeddingLoss()

    def forward(self, sentence_feature_real, sentence_feature_generated, target):
        #target = target.unsqueeze(0).expand([sentence_feature_real.size()[0], target.size()[0]])
        output = self.loss(sentence_feature_real.unsqueeze(0), sentence_feature_generated.unsqueeze(0), target.unsqueeze(0))
        return output
