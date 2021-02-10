import torch

class Filter(torch.nn.Module):
    
    def __init__(self,bias = False, kernel_size = 7, identity = True):
        
        super().__init__()
        
        if kernel_size % 2: # if kernel_size is odd
            padding = kernel_size // 2
        else: # if kernel_size is even
            padding = (torch.floor(torch.tensor(kernel_size / 2)).to(int).item(),
                       torch.ceil(torch.tensor(kernel_size / 2)).to(int).item() - 1)
        
        self.filter = torch.nn.Sequential(
                        torch.nn.ReflectionPad1d(padding),
                        torch.nn.Conv1d(in_channels = 1,
                                        out_channels = 1,
                                        kernel_size = kernel_size,
                                        stride = 1,
                                        padding = 0,
                                        bias = bias))
        
        if identity:
            for param in self.parameters():
                if param.ndim > 2: # only initialize conv weight not bias
                    torch.nn.init.dirac_(param)
        
    def forward(self,x):
        x = self.filter(x)
        return x

if __name__ == '__main__':
    
    filt = Filter(bias = False, kernel_size = 7, identity = True)
    
    batch_size = 8
    num_channels = 1
    sr = 16000
    length_sec = 1.5
    x = torch.rand(batch_size,num_channels,int(length_sec * sr))
    
    print('Input shape: {}'.format(tuple(x.shape)))
    
    y = filt(x)
    
    print('Filtered shape: {}'.format(tuple(y.shape)))
    