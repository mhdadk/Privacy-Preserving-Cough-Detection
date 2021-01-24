import warnings
warnings.simplefilter("ignore")

import torch
import torchaudio

class PTanh(torch.nn.Module):
    
    # PTanh(x) = a * tanh(b * x)
    
    def __init__(self,in_channels):
        
        super().__init__()
        
        # initialize a
        
        a = torch.full((in_channels,1,1), 1.7159)
        self.a = torch.nn.Parameter(a,requires_grad = True)
        
        # initialize b
        
        b = torch.full((in_channels,1,1), 2/3)
        self.b = torch.nn.Parameter(b,requires_grad = True)
    
    def forward(self,x):
        x = torch.multiply(x,self.a)
        x = torch.multiply(torch.nn.Tanh()(x),self.b)
        return x

# reconstruction network

batch_size = 8
num_channels = 1
length_sec = 1.5
sample_rate = 16000

# to count the total number of parameters

op_params = []

# example input

x1 = torch.rand(batch_size,num_channels,int(length_sec * sample_rate))
print('Input shape: {}'.format(tuple(x1.shape)))

# compute log Mel spectrogram
        
log = torchaudio.transforms.AmplitudeToDB()
mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = sample_rate,
                                                n_fft = 1024,
                                                n_mels = 128,
                                                hop_length = 64)
x = log(mel_spec(x1))
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

"""

# x = torch.multiply(torch.div(1,x.max(dim = 2)[0].unsqueeze(dim = -1) - 
#                                x.min(dim = 2)[0].unsqueeze(dim = -1)),
#                    x - x.min(dim = 2)[0].unsqueeze(dim = -1))

#%%

# encode

conv1 = torch.nn.Conv2d(in_channels = 1,
                        out_channels = 16,
                        kernel_size = 3,
                        stride = 1)

op_params.append(conv1)

y1 = conv1(x)

print('conv1 shape: {}'.format(tuple(y1.shape)))

act1 = PTanh(16)

y2 = act1(y1)

print('act1 shape: {}'.format(tuple(y2.shape)))

pool1 = torch.nn.Conv2d(in_channels = 16,
                        out_channels = 16,
                        kernel_size = 2,
                        stride = (1,2))

op_params.append(pool1)

y3 = pool1(y2)

print('pool1 shape: {}'.format(tuple(y3.shape)))

#%%

conv2 = torch.nn.Conv2d(in_channels = 16,
                        out_channels = 32,
                        kernel_size = 3,
                        stride = 1)

op_params.append(conv2)

y4 = conv2(y3)

print('conv2 shape: {}'.format(tuple(y4.shape)))

act2 = PTanh(32)

y5 = act2(y4)

print('act2 shape: {}'.format(tuple(y5.shape)))

pool2 = torch.nn.Conv2d(in_channels = 32,
                        out_channels = 32,
                        kernel_size = 2,
                        stride = (1,2))

op_params.append(pool2)

y6 = pool2(y5)

print('pool2 shape: {}'.format(tuple(y6.shape)))

#%%

# decode

us1 = torch.nn.Upsample(size = (130,130),
                        mode = 'nearest')

y7 = us1(y6)

print('us1 shape: {}'.format(tuple(y7.shape)))

conv3 = torch.nn.Conv2d(in_channels = 32,
                        out_channels = 16,
                        kernel_size = 3,
                        stride = 1)

op_params.append(conv3)

y8 = conv3(y7)

print('conv3 shape: {}'.format(tuple(y8.shape)))

act3 = PTanh(16)

y9 = act3(y8)

print('act3 shape: {}'.format(tuple(y9.shape)))

#%%

us2 = torch.nn.Upsample(size = (152,162),
                        mode = 'nearest')

y10 = us2(y9)

print('us2 shape: {}'.format(tuple(y10.shape)))

conv4 = torch.nn.Conv2d(in_channels = 16,
                        out_channels = 1,
                        kernel_size = 3,
                        stride = 1)

op_params.append(conv4)

y11 = conv4(y10)

print('conv4 shape: {}'.format(tuple(y11.shape)))

act4 = PTanh(1)

y12 = act4(y11)

print('act4 shape: {}'.format(tuple(y12.shape)))

#%%

# flatten to 1D signal

recon = torch.flatten(y12,start_dim=2)

# compute total number of parameters

import functools, operator

num_params = 0

for op in op_params:
    num_params += functools.reduce(operator.mul,op.weight.shape) + \
                  functools.reduce(operator.mul,op.bias.shape)
