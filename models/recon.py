import torch
from collections import OrderedDict

class PTanh(torch.nn.Module):
    
    # PTanh(x) = a * tanh(b * x)
    
    def __init__(self,num_parameters):
        
        super().__init__()
        
        # initialize a
        
        a = torch.full((num_parameters,1), 1.7159)
        # torch.nn.init.normal_(a)
        self.a = torch.nn.Parameter(a,requires_grad = True)
        
        # initialize b
        
        b = torch.full((num_parameters,1), 2/3)
        # torch.nn.init.normal_(b)
        self.b = torch.nn.Parameter(b,requires_grad = True)
    
    def forward(self,x):
        x = torch.multiply(x,self.a)
        x = torch.multiply(torch.nn.Tanh()(x),self.b)
        return x

class Autoencoder(torch.nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        # encoder --------------------------------------------------------
        
        encoder = OrderedDict()
        
        # first layer
        
        encoder['conv1'] = torch.nn.Conv1d(in_channels = 1,
                                           out_channels = 16,
                                           kernel_size = 3,
                                           stride = 1)
        
        encoder['act1'] = PTanh(16)
        
        encoder['out1'] = torch.nn.Conv1d(in_channels = 16,
                                          out_channels = 16,
                                          kernel_size = 2,
                                          stride = 2)
        
        # second layer
        
        encoder['conv2'] = torch.nn.Conv1d(in_channels = 16,
                                           out_channels = 32,
                                           kernel_size = 5,#3,
                                           stride = 1)
        
        encoder['act2'] = PTanh(32)
        
        encoder['out2'] = torch.nn.Conv1d(in_channels = 32,
                                          out_channels = 32,
                                          kernel_size = 2,
                                          stride = 2)
        
        # third layer
        
        encoder['conv3'] = torch.nn.Conv1d(in_channels = 32,
                                           out_channels = 64,
                                           kernel_size = 7,#3,
                                           stride = 1)
        
        encoder['act3'] = PTanh(64)
        
        encoder['out3'] = torch.nn.Conv1d(in_channels = 64,
                                          out_channels = 64,
                                          kernel_size = 2,
                                          stride = 2)
        
        self.encoder = torch.nn.Sequential(encoder)
        
        # decoder --------------------------------------------------------
        
        decoder = OrderedDict()
        
        # fourth layer
        
        decoder['us4'] = torch.nn.Upsample(scale_factor = 2,
                                           mode = 'nearest')
        
        decoder['conv4'] = torch.nn.Conv1d(in_channels = 64,
                                           out_channels = 32,
                                           kernel_size = 7,#3,
                                           stride = 1)
        
        decoder['out4'] = PTanh(32)
        
        # fifth layer
        
        decoder['us5'] = torch.nn.Upsample(scale_factor = 2,
                                           mode = 'nearest')
        
        decoder['conv5'] = torch.nn.Conv1d(in_channels = 32,
                                           out_channels = 16,
                                           kernel_size = 5,#3,
                                           stride = 1)
        
        decoder['out5'] = PTanh(16)
        
        # sixth layer
        
        decoder['us6'] = torch.nn.Upsample(size = 3006,
                                           mode = 'nearest')
        
        decoder['act1'] = PTanh(16)
        
        decoder['out6'] = torch.nn.Conv1d(in_channels = 16,
                                          out_channels = 1,
                                          kernel_size = 7,#3,
                                          stride = 1)
        
        self.decoder = torch.nn.Sequential(decoder)
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    
    net = Autoencoder()
    
    x = torch.rand(8,1,3000)
    
    # zero mean and unit variance TODO: try 0.5 mean instead

    x = torch.div((x - x.mean(dim = 2).unsqueeze(dim = -1) + 0.5),
                   x.std(dim = 2).unsqueeze(dim = -1))
    
    """
    scale each signal to be between 0 and 1. This is equivalent to:
    
    y = {x / (x.max - x.min)} - {x.min / (x.max - x.min)}
      = (1 / x.max - x.min) * (x - x.min)
    
    this is only necessary if the negative log-likelihood function for a
    Bernoulli distribution (binary cross-entropy) will be used as a loss
    function
    
    """
    
    x = torch.multiply(torch.div(1,x.max(dim = 2)[0].unsqueeze(dim = -1) - 
                                   x.min(dim = 2)[0].unsqueeze(dim = -1)),
                       x - x.min(dim = 2)[0].unsqueeze(dim = -1))
    
    # compute reconstruction
    
    x_hat = net(x)