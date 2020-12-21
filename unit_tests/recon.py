import torch

class PTanh(torch.nn.Module):
    
    # PTanh(x) = a * tanh(b * x)
    
    def __init__(self,num_parameters):
        
        super().__init__()
        
        # initialize a
        
        a = torch.full((num_parameters,1), 1.7159)
        self.a = torch.nn.Parameter(a,requires_grad = True)
        
        # initialize b
        
        b = torch.full((num_parameters,1), 2/3)
        self.b = torch.nn.Parameter(b,requires_grad = True)
    
    def forward(self,x):
        x = torch.multiply(x,self.a)
        x = torch.multiply(torch.nn.Tanh()(x),self.b)
        return x

# reconstruction network

batch_size = 8
num_channels = 1
length_sec = 1.5
sample_rate = 2000

# to count the total number of parameters

op_params = []

# example input

x = torch.rand(batch_size,num_channels,int(length_sec * sample_rate))

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

#%%

# encode

conv1 = torch.nn.Conv1d(in_channels = 1,
                        out_channels = 16,
                        kernel_size = 3,
                        stride = 1)

op_params.append(conv1)

act1 = PTanh(16)

y1 = act1(conv1(x))

pool1 = torch.nn.Conv1d(in_channels = 16,
                        out_channels = 16,
                        kernel_size = 2,
                        stride = 2)

op_params.append(pool1)

y2 = pool1(y1)

#%%

conv2 = torch.nn.Conv1d(in_channels = 16,#64,
                        out_channels = 32,#56,
                        kernel_size = 3,#5,#16,
                        stride = 1)

op_params.append(conv2)

y3 = activation(conv2(y2))

pool2 = torch.nn.Conv1d(in_channels = 32,
                        out_channels = 32,
                        kernel_size = 2,
                        stride = 2)

op_params.append(pool2)

y4 = pool2(y3)

#%%

conv3 = torch.nn.Conv1d(in_channels = 32,#64,
                        out_channels = 64,#56,
                        kernel_size = 3,#7,#16,
                        stride = 1)

op_params.append(conv3)

y5 = activation(conv3(y4))

pool3 = torch.nn.Conv1d(in_channels = 64,
                        out_channels = 64,
                        kernel_size = 2,
                        stride = 2)

op_params.append(pool3)

y6 = pool3(y5)

#%%

# decode

us1 = torch.nn.Upsample(scale_factor = 2,
                        mode = 'nearest')

y7 = us1(y6)

conv4 = torch.nn.Conv1d(in_channels = 64,
                        out_channels = 32,
                        kernel_size = 3,#7,
                        stride = 1)

op_params.append(conv4)

y8 = activation(conv4(y7))

#%%

us2 = torch.nn.Upsample(scale_factor = 2,
                        mode = 'nearest')

y9 = us2(y8)

conv5 = torch.nn.Conv1d(in_channels = 32,
                        out_channels = 16,
                        kernel_size = 3,#5,
                        stride = 1)

op_params.append(conv5)

y10 = activation(conv5(y9))

#%%

us3 = torch.nn.Upsample(size = 3002,
                        mode = 'nearest')

y11 = us3(y10)

conv6 = torch.nn.Conv1d(in_channels = 16,
                        out_channels = 1,
                        kernel_size = 3,#7,#3,
                        stride = 1)

op_params.append(conv6)

y12 = activation(conv6(y11))

#%%

# compute total number of parameters

import functools, operator

num_params = 0

for op in op_params:
    num_params += functools.reduce(operator.mul,op.weight.shape) + \
                  functools.reduce(operator.mul,op.bias.shape)
