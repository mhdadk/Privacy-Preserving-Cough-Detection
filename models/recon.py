import torch

class PTanh(torch.nn.Module):
    
    # PTanh(x) = a * tanh(b * x)
    
    def __init__(self,in_channels):
        
        super().__init__()
        
        # need this for __repr__ method
        
        self.in_channels = in_channels
        
        # initialize a
        
        a = torch.full((in_channels,1), 1.7159)
        self.a = torch.nn.Parameter(a,requires_grad = True)
        
        # initialize b
        
        b = torch.full((in_channels,1), 2/3)
        self.b = torch.nn.Parameter(b,requires_grad = True)
    
    def __repr__(self):
        return 'PTanh(' + str(self.in_channels) + ')'
    
    def forward(self,x):
        x = torch.multiply(x,self.a)
        x = torch.multiply(torch.nn.Tanh()(x),self.b)
        return x

class Autoencoder(torch.nn.Module):
    
    def __init__(self,batch_norm = False):
        
        super().__init__()
        
        # encoder --------------------------------------------------------
        
        encoder = []
        
        # first layer
        
        layer = [torch.nn.Conv1d(in_channels = 1,
                                 out_channels = 16,
                                 kernel_size = 3,
                                 stride = 1,
                                 bias = True),
                 PTanh(16),
                 torch.nn.Conv1d(in_channels = 16,
                                 out_channels = 16,
                                 kernel_size = 2,
                                 stride = 2,
                                 bias = True)]
        
        encoder.extend(layer)
        
        if batch_norm:
            encoder.append(torch.nn.BatchNorm1d(num_features = 16,
                                                eps = 1e-08,
                                                momentum = 0.1,
                                                affine = True,
                                                track_running_stats = True))
        
        # second layer
        
        layer = [torch.nn.Conv1d(in_channels = 16,
                                 out_channels = 32,
                                 kernel_size = 5,
                                 stride = 1,
                                 bias = True),
                 PTanh(32),
                 torch.nn.Conv1d(in_channels = 32,
                                 out_channels = 32,
                                 kernel_size = 2,
                                 stride = 2,
                                 bias = True)]
        
        encoder.extend(layer)
        
        if batch_norm:
            encoder.append(torch.nn.BatchNorm1d(num_features = 32,
                                                eps = 1e-08,
                                                momentum = 0.1,
                                                affine = True,
                                                track_running_stats = True))
        
        # third layer
        
        layer = [torch.nn.Conv1d(in_channels = 32,
                                 out_channels = 64,
                                 kernel_size = 7,
                                 stride = 1,
                                 bias = True),
                 PTanh(64),
                 torch.nn.Conv1d(in_channels = 64,
                                 out_channels = 64,
                                 kernel_size = 2,
                                 stride = 2,
                                 bias = True)]
        
        encoder.extend(layer)
        
        if batch_norm:
            encoder.append(torch.nn.BatchNorm1d(num_features = 64,
                                                eps = 1e-08,
                                                momentum = 0.1,
                                                affine = True,
                                                track_running_stats = True))
        
        self.encoder = torch.nn.Sequential(*encoder)
        
        # decoder --------------------------------------------------------
        
        decoder = []
        
        # fourth layer
        
        layer = [torch.nn.Upsample(scale_factor = 2,
                                   mode = 'nearest'),
                 torch.nn.Conv1d(in_channels = 64,
                                 out_channels = 32,
                                 kernel_size = 7,
                                 stride = 1,
                                 bias = True),
                 PTanh(32)]
        
        decoder.extend(layer)
        
        if batch_norm:
            encoder.append(torch.nn.BatchNorm1d(num_features = 32,
                                                eps = 1e-08,
                                                momentum = 0.1,
                                                affine = True,
                                                track_running_stats = True))
        
        # fifth layer
        
        layer = [torch.nn.Upsample(scale_factor = 2,
                                   mode = 'nearest'),
                 torch.nn.Conv1d(in_channels = 32,
                                 out_channels = 16,
                                 kernel_size = 5,
                                 stride = 1,
                                 bias = True),
                 PTanh(16)]
        
        decoder.extend(layer)
        
        if batch_norm:
            encoder.append(torch.nn.BatchNorm1d(num_features = 16,
                                                eps = 1e-08,
                                                momentum = 0.1,
                                                affine = True,
                                                track_running_stats = True))
        
        # sixth layer
        
        layer = [torch.nn.Upsample(size = 3006,
                                   mode = 'nearest'),
                 PTanh(16),
                 torch.nn.Conv1d(in_channels = 16,
                                 out_channels = 1,
                                 kernel_size = 7,
                                 stride = 1,
                                 bias = True)]
        
        decoder.extend(layer)
        
        self.decoder = torch.nn.Sequential(*decoder)
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    
    from torchsummary import summary
    
    net = Autoencoder(batch_norm = True)
    net.eval()
    
    batch_size = 8
    num_channels = 1
    window_length = 1.5 # seconds
    sample_rate = 2000
    
    x = torch.rand(batch_size,num_channels,int(window_length * sample_rate))
    
    print('Input shape: {}'.format(tuple(x.shape)))
    
    # zero mean and unit variance TODO: try 0.5 mean instead

    x = torch.div((x - x.mean(dim = 2).unsqueeze(dim = -1)),
                   x.std(dim = 2).unsqueeze(dim = -1))
    
    print('Normalized shape: {}'.format(tuple(x.shape)))
    print('Normalized mean: {}'.format(x.mean()))
    print('Normalized variance: {}'.format(x.var()))
    
    """
    scale each signal to be between 0 and 1. This is equivalent to:
    
    y = {x / (x.max - x.min)} - {x.min / (x.max - x.min)}
      = (1 / x.max - x.min) * (x - x.min)
    
    this is only necessary if the negative log-likelihood function for a
    Bernoulli distribution (binary cross-entropy) will be used as a loss
    function
    
    """
    
    # x = torch.multiply(torch.div(1,x.max(dim = 2)[0].unsqueeze(dim = -1) - 
    #                                x.min(dim = 2)[0].unsqueeze(dim = -1)),
    #                    x - x.min(dim = 2)[0].unsqueeze(dim = -1))
    
    # summary of net
    
    summary(net,tuple(x.shape[1:]))
    