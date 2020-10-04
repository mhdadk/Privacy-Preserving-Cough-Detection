import torch
from FENet import FENet

class Disc(torch.nn.Module):

    def __init__(self):
        super(Disc, self).__init__()
        param_path = 'mx-h64-1024_0d3-1.17.pkl'
        self.feature_extractor = FENet(param_path)
        # freeze weights of feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.fc1 = torch.nn.Linear(
            in_features = self.feature_extractor.stage7[0].out_channels,
            out_features = 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc1(x)
        return x

net = Disc()