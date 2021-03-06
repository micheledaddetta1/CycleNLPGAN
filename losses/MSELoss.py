import torch
from torch import nn, Tensor



class MSELoss(nn.Module):
    """
    Computes the MSE loss between the computed sentence embedding and a target sentence embedding
    """
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_feature_real, sentence_feature_generated, target):
        return self.loss_fct(sentence_feature_generated,sentence_feature_real)

