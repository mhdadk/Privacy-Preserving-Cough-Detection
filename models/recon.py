import torch

class Autoencoder(torch.nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.filt = torch.nn.Conv1d(in_channels = 1,
                                    out_channels = 1,
                                    kernel_size = 40,
                                    stride = 1)
        
        pool = torch.nn.MaxPool1d(kernel_size = 2)
        
        conv1 = torch.nn.Conv1d(in_channels = 1,
                                out_channels = 1,
                                kernel_size = 40,
                                stride = 1)
        
        conv2 = torch.nn.Conv1d(in_channels = 1,
                                out_channels = 1,
                                kernel_size = 40,
                                stride = 1)
        
        self.encoder = torch.nn.Sequential(pool,
                                           conv1,
                                           pool,
                                           conv2,
                                           pool)
        
        upsample = torch.nn.Upsample(scale_factor = 3,
                                     mode = 'nearest')
        
        conv3 = torch.nn.Conv1d(in_channels = 1,
                                out_channels = 1,
                                kernel_size = 15,
                                stride = 1,
                                padding = 0)
        
        conv4 = torch.nn.Conv1d(in_channels = 1,
                                out_channels = 1,
                                kernel_size = 19,
                                stride = 1,
                                padding = 0)
        
        self.decoder = torch.nn.Sequential(upsample,
                                           conv3,
                                           upsample,
                                           conv4)
        
    def forward(self,x):
        
        x = self.filt(x)
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x

if __name__ == '__main__':
    
    net = Autoencoder()
    
    x = torch.rand(32,1,3000)
    
    y = net(x)