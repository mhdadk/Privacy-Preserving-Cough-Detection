import torch
import functools, operator

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

# encode

activation = torch.nn.SELU()

conv1 = torch.nn.Conv1d(in_channels = 1,
                        out_channels = 32,
                        kernel_size = 32,
                        stride = 1)

op_params.append(conv1)

y1 = activation(conv1(x))

pool1 = torch.nn.MaxPool1d(kernel_size = 2)

y2 = pool1(y1)

conv2 = torch.nn.Conv1d(in_channels = 32,
                        out_channels = 56,
                        kernel_size = 16,
                        stride = 1)

op_params.append(conv2)

y3 = activation(conv2(y2))

y4 = pool1(y3)

# decode

us1 = torch.nn.Upsample(scale_factor = 2,
                        mode = 'nearest')

y5 = us1(y4)

conv3 = torch.nn.Conv1d(in_channels = 56,
                        out_channels = 32,
                        kernel_size = 16,
                        stride = 1)

op_params.append(conv3)

y6 = activation(conv3(y5))

us2 = torch.nn.Upsample(size = 3031,
                        mode = 'nearest')

y7 = us2(y6)

conv4 = torch.nn.Conv1d(in_channels = 32,
                        out_channels = 1,
                        kernel_size = 32,
                        stride = 1)

op_params.append(conv4)

y8 = torch.sigmoid(conv4(y7))

# compute total number of parameters

num_params = 0

for op in op_params:
    num_params += functools.reduce(operator.mul,op.weight.shape) + \
                  functools.reduce(operator.mul,op.bias.shape)
