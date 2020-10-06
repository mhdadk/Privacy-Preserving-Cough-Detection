import torch

"""
FENet outputs (C,H,W):
Input = 1 x 256 x 254
Stage1 = 16 x 128 x 127
Stage2 = 32 x 64 x 63
Stage3 = 64 x 32 x 31
Stage4 = 128 x 16 x 15
Stage5 = 256 x 8 x 7
Stage6 = 512 x 4 x 3
Stage7 = 1024 x 3 x 2
"""

x = torch.randn(1,1024,3,2)

kernels = [((3,3),(3,3)),
           ((3,3),(3,3)),
           ((3,3),(3,2)),
           ((3,3),(3,3)),
           ((3,3),(3,3)),
           ((3,3),(3,3))]

strides = [((2,2),(1,1)),
           ((2,3),(1,1)),
           ((2,2),(1,1)),
           ((2,2),(1,1)),
           ((2,2),(1,1)),
           ((2,2),(1,1))]

paddings = [((0,0),(0,0)),
            ((0,0),(0,0)),
            ((0,0),(0,0)),
            ((0,0),(0,2)),
            ((0,0),(0,1)),
            ((0,0),(0,1))]

stages2 = []
in_channels = 1024
activation = torch.nn.ReLU()

for kernel,stride,padding in zip(kernels,strides,paddings):

    # part 1    

    conv1 = torch.nn.ConvTranspose2d(in_channels = in_channels,
                                     out_channels = int(in_channels/2),
                                     kernel_size = kernel[0],
                                     stride = stride[0],
                                     padding = padding[0])
    
    batch_norm1 = torch.nn.BatchNorm2d(int(in_channels/2))
    
    # part 2
    
    conv2 = torch.nn.Conv2d(in_channels = int(in_channels/2),
                            out_channels = int(in_channels/2),
                            kernel_size = kernel[1],
                            stride = stride[1],
                            padding = padding[1])
    
    batch_norm2 = torch.nn.BatchNorm2d(int(in_channels/2))
    
    # stage
    
    stages2 += torch.nn.Sequential(conv1,
                                   batch_norm1,
                                   activation,
                                   conv2,
                                   batch_norm2,
                                   activation)
    
    in_channels = int(in_channels/2)

# stage 7

# part 1    

conv1 = torch.nn.ConvTranspose2d(in_channels = in_channels,
                                 out_channels = 1,
                                 kernel_size = (3,3),
                                 stride = (2,2),
                                 padding = (0,0))

batch_norm1 = torch.nn.BatchNorm2d(1)

# part 2

zero_pad = torch.nn.ZeroPad2d(padding = (0,1,0,0))
conv2 = torch.nn.Conv2d(in_channels = 1,
                        out_channels = 1,
                        kernel_size = (4,3),
                        stride = (1,1),
                        padding = (0,0))

batch_norm2 = torch.nn.BatchNorm2d(1)

# stage

stages2 += torch.nn.Sequential(conv1,
                               batch_norm1,
                               activation,
                               zero_pad,
                               conv2,
                               batch_norm2,
                               activation)

# build net

net = torch.nn.Sequential(*stages2)

# test net

y = net(x)
