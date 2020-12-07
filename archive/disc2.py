import torch

class Disc(torch.nn.Module):

    def __init__(self):
        super(Disc, self).__init__()
        self.fc1 = torch.nn.Linear(in_features = 1024,
                                   out_features = 1)

    def forward(self, x):
        x = self.fc1(x)
        return x

# net = Disc()