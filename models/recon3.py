import torch

class PTanh(torch.nn.Module):
    
    # PTanh(x) = a * tanh(b * x)
    
    def __init__(self,in_channels):
        
        super().__init__()
        
        # need this for __repr__ method
        
        self.in_channels = in_channels
        
        # initialize a
        
        a = torch.full((in_channels,1,1), 1.7159)
        self.a = torch.nn.Parameter(a,requires_grad = True)
        
        # initialize b
        
        b = torch.full((in_channels,1,1), 2/3)
        self.b = torch.nn.Parameter(b,requires_grad = True)
    
    def __repr__(self):
        return 'PTanh(' + str(self.in_channels) + ')'
    
    def forward(self,x):
        x = torch.multiply(x,self.a)
        x = torch.multiply(torch.nn.Tanh()(x),self.b)
        return x

class Autoencoder(torch.nn.Module):
    
    def __init__(self,inst_norm = True, num_channels = 16):
        
        super().__init__()
        
        self.inst_norm = inst_norm
        
        self.encoder = torch.nn.Conv2d(in_channels = 1,
                                       out_channels = num_channels,
                                       kernel_size = 3,
                                       stride = 1,
                                       padding = 1,
                                       padding_mode = 'replicate',
                                       bias = True)
        
        self.normalization = torch.nn.InstanceNorm2d(num_features = num_channels,
                                                     eps = 1e-8,
                                                     momentum = 0.1,
                                                     affine = True,
                                                     track_running_stats = True)
        
        self.decoder = torch.nn.Conv2d(in_channels = num_channels,
                                       out_channels = 1,
                                       kernel_size = 3,
                                       stride = 1,
                                       padding = 1,
                                       padding_mode = 'replicate',
                                       bias = True)
        
        self.activation = torch.nn.Sigmoid()
        
    def forward(self,x):
        x = self.encoder(x)
        if self.inst_norm:
            x = self.normalization(x)
        x = self.decoder(x)
        x = self.activation(x)
        return x

if __name__ == '__main__':
    
    import torchaudio
    from torchsummary import summary
    
    net = Autoencoder()
    
    batch_size = 8
    num_channels = 1
    sr = 16000
    length_sec = 1.5
    x = torch.rand(batch_size,num_channels,int(length_sec * sr))
    
    print('Input shape: {}'.format(tuple(x.shape)))
    
    win_length_sec = 0.012
    win_length = int(sr * win_length_sec)
    # need 50% overlap to satisfy constant-overlap-add constraint to allow
    # for perfect reconstruction using inverse STFT
    hop_length = int(sr * win_length_sec / 2)
    
    spec = torchaudio.transforms.Spectrogram(n_fft = 512,
                                             win_length = win_length,
                                             hop_length = hop_length,
                                             pad = 0,
                                             window_fn = torch.hann_window,
                                             power = None,
                                             normalized = False,
                                             wkwargs = None)
    x = spec(x)
    
    print('Spectrogram shape: {}'.format(tuple(x.shape)))
    
    x,phase = torchaudio.functional.magphase(x)
    
    x = x / x.amax(dim=(2,3))[(..., ) + (None, ) * 2]
    
    # y = net(x)
    
    # # zero mean and unit variance. The expression [(..., ) + (None, ) * 2] is
    # # used to unsqueeze 2 dimensions at the end of x.mean(dim = (2,3))
    
    # x = torch.div(x - x.mean(dim = (2,3))[(..., ) + (None, ) * 2],
    #               x.std(dim = (2,3))[(..., ) + (None, ) * 2])
    # print('Normalized shape: {}'.format(tuple(x.shape)))
    # print('Normalized mean: {}'.format(x.mean()))
    # print('Normalized variance: {}'.format(x.var()))
    
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
    