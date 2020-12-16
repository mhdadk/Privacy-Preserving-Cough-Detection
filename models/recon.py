import torch

class Autoencoder(torch.nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        # encoder
        
        conv1 = torch.nn.Conv1d(in_channels = 1,
                                out_channels = 32,
                                kernel_size = 32,
                                stride = 1)
        
        activation1 = torch.nn.SELU()
        
        pool1 = torch.nn.MaxPool1d(kernel_size = 2)
        
        conv2 = torch.nn.Conv1d(in_channels = 32,
                                out_channels = 56,
                                kernel_size = 16,
                                stride = 1)
        
        self.encoder = torch.nn.Sequential(conv1,
                                           activation1,
                                           pool1,
                                           conv2,
                                           activation1,
                                           pool1)
        
        # decoder
        
        upsample1 = torch.nn.Upsample(scale_factor = 2,
                                      mode = 'nearest')
        
        conv3 = torch.nn.Conv1d(in_channels = 56,
                                out_channels = 32,
                                kernel_size = 16,
                                stride = 1)
        
        upsample2 = torch.nn.Upsample(size = 3031,
                                      mode = 'nearest')
        
        conv4 = torch.nn.Conv1d(in_channels = 32,
                                out_channels = 1,
                                kernel_size = 32,
                                stride = 1)
        
        activation2 = torch.nn.Sigmoid()
        
        self.decoder = torch.nn.Sequential(upsample1,
                                           conv3,
                                           activation1,
                                           upsample2,
                                           conv4,
                                           activation2)
        
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