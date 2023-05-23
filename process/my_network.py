import torch.nn as nn
from torch.nn import functional as F
import torch

class network(nn.Module):
    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        

    def forward(self, input):
        x = self.feature(input)
        x = self.classifier(x)
        return x