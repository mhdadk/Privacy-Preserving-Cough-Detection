import torch

class Filter(torch.nn.Module):
    
    def __init__(self,identity = True):
        
        super().__init__()
        
        self.filter = torch.nn.Conv1d(in_channels = 1,
                                      out_channels = 1,
                                      kernel_size = 7,
                                      stride = 1,
                                      padding = 3,
                                      bias = False)
        
        if identity:
            for param in self.parameters():
                torch.nn.init.dirac_(param)
        
    def forward(self,x):
        x = self.filter(x)
        return x

if __name__ == '__main__':
    
    import torchaudio
    from torchsummary import summary
    
    filt = Filter()
    
    batch_size = 8
    num_channels = 1
    sr = 16000
    length_sec = 1.5
    x = torch.rand(batch_size,num_channels,int(length_sec * sr))
    
    print('Input shape: {}'.format(tuple(x.shape)))
    
    y = filt(x)
    
    print('Filtered shape: {}'.format(tuple(y.shape)))
    
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
    y_spec = spec(y)
    
    print('Spectrogram shape: {}'.format(tuple(y_spec.shape)))
    
    y_mag,phase = torchaudio.functional.magphase(y_spec)
    
    y_mag = y_mag / y_mag.amax(dim=(2,3))[(..., ) + (None, ) * 2]
    
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
    
    # summary(net,tuple(x.shape[1:]))
    