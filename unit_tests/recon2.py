import warnings
warnings.simplefilter("ignore")

import torch
import torchaudio

from torchsummary import summary

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
    
    def __init__(self):
        
        super().__init__()
        
        # encoder --------------------------------------------------------
        
        encoder = []
        
        first_layer = [torch.nn.Conv2d(in_channels = 1,
                                       out_channels = 32,
                                       kernel_size = 3,
                                       stride = 1),
                       PTanh(32),
                       torch.nn.Conv2d(in_channels = 32,
                                       out_channels = 32,
                                       kernel_size = 2,
                                       stride = (1,2))]
        
        encoder.extend(first_layer)
        
        second_layer = [torch.nn.Conv2d(in_channels = 32,
                                       out_channels = 64,
                                       kernel_size = 3,
                                       stride = 1),
                       PTanh(64),
                       torch.nn.Conv2d(in_channels = 64,
                                       out_channels = 64,
                                       kernel_size = 2,
                                       stride = 2)]
        
        encoder.extend(second_layer)
        
        self.encoder = torch.nn.Sequential(*encoder)
        
        # decoder --------------------------------------------------------
        
        decoder = []
        
        third_layer = [torch.nn.Upsample(size = (130,130),
                                         mode = 'nearest'),
                       torch.nn.Conv2d(in_channels = 64,
                                       out_channels = 32,
                                       kernel_size = 3,
                                       stride = 1),
                       PTanh(32)]
        
        decoder.extend(third_layer)
        
        fourth_layer = [torch.nn.Upsample(size = (152,162),
                                          mode = 'nearest'),
                        torch.nn.Conv2d(in_channels = 32,
                                        out_channels = 1,
                                        kernel_size = 3,
                                        stride = 1),
                        PTanh(1)]
        
        decoder.extend(fourth_layer)
        
        self.decoder = torch.nn.Sequential(*decoder)
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    
    net = Autoencoder()
    
    batch_size = 8
    num_channels = 1
    length_sec = 1.5
    sample_rate = 16000
    
    x = torch.rand(batch_size,num_channels,int(length_sec * sample_rate))
    
    print('Input shape: {}'.format(tuple(x.shape)))
    
    log = torchaudio.transforms.AmplitudeToDB()
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = sample_rate,
                                                    n_fft = 1024,
                                                    n_mels = 128,
                                                    hop_length = 64)
    x = log(mel_spec(x))
    
    print('Log Mel-scale spectrogram shape: {}'.format(tuple(x.shape)))
    
    # zero mean and unit variance. The expression [(..., ) + (None, ) * 2] is
    # used to unsqueeze 2 dimensions at the end of x.mean(dim = (2,3))
    
    x = torch.div(x - x.mean(dim = (2,3))[(..., ) + (None, ) * 2],
                  x.std(dim = (2,3))[(..., ) + (None, ) * 2])
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
    
    # get shape of output
    
    y = net(x)
    recon = torch.flatten(y,start_dim=2)
    print('Output shape: {}'.format(tuple(recon.shape)))