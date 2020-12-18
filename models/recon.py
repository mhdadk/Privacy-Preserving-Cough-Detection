import torch
from collections import OrderedDict

class Autoencoder(torch.nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        # encoder
        
        encoder = OrderedDict()
        
        encoder['conv1'] = torch.nn.Conv1d(in_channels = 1,
                                           out_channels = 32,
                                           kernel_size = 32,
                                           stride = 1)
        
        encoder['act1'] = torch.nn.SELU()
        
        encoder['out1'] = torch.nn.MaxPool1d(kernel_size = 2)
        
        encoder['conv2'] = torch.nn.Conv1d(in_channels = 32,
                                           out_channels = 56,
                                           kernel_size = 16,
                                           stride = 1)
        
        encoder['act2'] = torch.nn.SELU()
        
        encoder['out2'] = torch.nn.MaxPool1d(kernel_size = 2)
        
        self.encoder = torch.nn.Sequential(encoder)
        
        # decoder
        
        decoder = OrderedDict()
        
        decoder['us1'] = torch.nn.Upsample(scale_factor = 2,
                                           mode = 'nearest')
        
        decoder['conv3'] = torch.nn.Conv1d(in_channels = 56,
                                           out_channels = 32,
                                           kernel_size = 16,
                                           stride = 1)
        
        decoder['out3'] = torch.nn.SELU()
        
        decoder['us2'] = torch.nn.Upsample(size = 3031,
                                      mode = 'nearest')
        
        decoder['conv4'] = torch.nn.Conv1d(in_channels = 32,
                                           out_channels = 1,
                                           kernel_size = 32,
                                           stride = 1)
        
        decoder['out4'] = torch.nn.Sigmoid()
        
        self.decoder = torch.nn.Sequential(decoder)
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    
    net = Autoencoder()
    
    x = torch.rand(8,1,3000)
    
    # zero mean and unit variance

    x = torch.div((x - x.mean(dim = 2).unsqueeze(dim = -1)),
                   x.std(dim = 2).unsqueeze(dim = -1))
    
    """
    scale each signal to be between 0 and 1. This is equivalent to:
    
    y = {x / (x.max - x.min)} - {x.min / (x.max - x.min)}
      = (1 / x.max - x.min) * (x - x.min)
    
    """
    
    x = torch.multiply(torch.div(1,x.max(dim = 2)[0].unsqueeze(dim = -1) - 
                                   x.min(dim = 2)[0].unsqueeze(dim = -1)),
                       x - x.min(dim = 2)[0].unsqueeze(dim = -1))
    
    # compute reconstruction
    
    x_hat = net(x)