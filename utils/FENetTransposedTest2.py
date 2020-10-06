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

x = torch.randn(1,1024,1,1)
in_channels = x.shape[1]
stages = []

# stage 1 part 1, inverse of average pool

conv1 = torch.nn.ConvTranspose2d(in_channels = in_channels,
                                 out_channels = in_channels,
                                 kernel_size = (3,2),
                                 stride = 2,
                                 padding = 0)

y1 = conv1(x)

# stage 1 part 2

conv2 = torch.nn.Conv2d(in_channels = in_channels,
                        out_channels = in_channels,
                        kernel_size = (3,2),
                        stride = 1,
                        padding = (2,1))

y2 = conv2(x)

batch_norm1 = torch.nn.BatchNorm2d(in_channels)

y2 = batch_norm1(y2)

activation = torch.nn.ReLU()

y2 = activation(y2)

# stage 1

stages += torch.nn.Sequential(conv1,
                              conv2,
                              batch_norm1,
                              activation)

# stage 2 part 1

conv1 = torch.nn.ConvTranspose2d(in_channels = y2.shape[1],
                                 out_channels = int(y2.shape[1]/2),
                                 kernel_size = (3,3),
                                 stride = (2,2),
                                 padding = 0)
y1 = conv1(y2)

batch_norm1 = torch.nn.BatchNorm2d(int(y2.shape[1]/2))
y1 = batch_norm1(y1)

y1 = activation(y1)

# stage 2 part 2

conv2 = torch.nn.Conv2d(in_channels = y1.shape[1],
                        out_channels = y1.shape[1],
                        kernel_size = 3,
                        stride = 1,
                        padding = 0)
y2 = conv2(y1)

batch_norm2 = torch.nn.BatchNorm2d(y1.shape[1])
y2 = batch_norm2(y2)

y2 = activation(y2)

# stage 2

stages += torch.nn.Sequential(conv1,
                              batch_norm1,
                              activation,
                              conv2,
                              batch_norm2,
                              activation)

# stage 3 part 1

conv1 = torch.nn.ConvTranspose2d(in_channels = y2.shape[1],
                                 out_channels = int(y2.shape[1]/2),
                                 kernel_size = 3,
                                 stride = 2,
                                 padding = 0)
y1 = conv1(y2)

batch_norm1 = torch.nn.BatchNorm2d(int(y2.shape[1]/2))
y1 = batch_norm1(y1)

y1 = activation(y1)

# stage 3 part 2

conv2 = torch.nn.Conv2d(in_channels = y1.shape[1],
                        out_channels = y1.shape[1],
                        kernel_size = (3,2),
                        stride = 1,
                        padding = 0)
y2 = conv2(y1)

batch_norm2 = torch.nn.BatchNorm2d(y1.shape[1])
y2 = batch_norm2(y2)

y2 = activation(y2)

# stage 3

stages += torch.nn.Sequential(conv1,
                              batch_norm1,
                              activation,
                              conv2,
                              batch_norm2,
                              activation)

# stage 4 part 1

conv1 = torch.nn.ConvTranspose2d(in_channels = y2.shape[1],
                                 out_channels = int(y2.shape[1]/2),
                                 kernel_size = 3,
                                 stride = 2,
                                 padding = 0)
y1 = conv1(y2)

batch_norm1 = torch.nn.BatchNorm2d(int(y2.shape[1]/2))
y1 = batch_norm1(y1)

y1 = activation(y1)

# stage 4 part 2

conv2 = torch.nn.Conv2d(in_channels = y1.shape[1],
                        out_channels = y1.shape[1],
                        kernel_size = 3,
                        stride = 1,
                        padding = (0,2))
y2 = conv2(y1)

batch_norm2 = torch.nn.BatchNorm2d(y1.shape[1])
y2 = batch_norm2(y2)

y2 = activation(y2)

# stage 4

stages += torch.nn.Sequential(conv1,
                              batch_norm1,
                              activation,
                              conv2,
                              batch_norm2,
                              activation)

# stage 5 part 1

conv1 = torch.nn.ConvTranspose2d(in_channels = y2.shape[1],
                                 out_channels = int(y2.shape[1]/2),
                                 kernel_size = 3,
                                 stride = 2,
                                 padding = 0)
y1 = conv1(y2)

batch_norm1 = torch.nn.BatchNorm2d(int(y2.shape[1]/2))
y1 = batch_norm1(y1)

y1 = activation(y1)

# stage 5 part 2

conv2 = torch.nn.Conv2d(in_channels = y1.shape[1],
                        out_channels = y1.shape[1],
                        kernel_size = 3,
                        stride = 1,
                        padding = (0,1))
y2 = conv2(y1)

batch_norm2 = torch.nn.BatchNorm2d(y1.shape[1])
y2 = batch_norm2(y2)

y2 = activation(y2)

# stage 5

stages += torch.nn.Sequential(conv1,
                              batch_norm1,
                              activation,
                              conv2,
                              batch_norm2,
                              activation)

# stage 6 part 1

conv1 = torch.nn.ConvTranspose2d(in_channels = y2.shape[1],
                                 out_channels = int(y2.shape[1]/2),
                                 kernel_size = 3,
                                 stride = 2,
                                 padding = 0)
y1 = conv1(y2)

batch_norm1 = torch.nn.BatchNorm2d(int(y2.shape[1]/2))
y1 = batch_norm1(y1)

y1 = activation(y1)

# stage 6 part 2

conv2 = torch.nn.Conv2d(in_channels = y1.shape[1],
                        out_channels = y1.shape[1],
                        kernel_size = 3,
                        stride = 1,
                        padding = (0,1))
y2 = conv2(y1)

batch_norm2 = torch.nn.BatchNorm2d(y1.shape[1])
y2 = batch_norm2(y2)

y2 = activation(y2)

# stage 6

stages += torch.nn.Sequential(conv1,
                              batch_norm1,
                              activation,
                              conv2,
                              batch_norm2,
                              activation)

# stage 7 part 1

conv1 = torch.nn.ConvTranspose2d(in_channels = y2.shape[1],
                                 out_channels = int(y2.shape[1]/2),
                                 kernel_size = 3,
                                 stride = 2,
                                 padding = 0)
y1 = conv1(y2)

batch_norm1 = torch.nn.BatchNorm2d(int(y2.shape[1]/2))
y1 = batch_norm1(y1)

y1 = activation(y1)

# stage 7 part 2

conv2 = torch.nn.Conv2d(in_channels = y1.shape[1],
                        out_channels = y1.shape[1],
                        kernel_size = 3,
                        stride = 1,
                        padding = (0,1))
y2 = conv2(y1)

batch_norm2 = torch.nn.BatchNorm2d(y1.shape[1])
y2 = batch_norm2(y2)

y2 = activation(y2)

# stage 7

stages += torch.nn.Sequential(conv1,
                              batch_norm1,
                              activation,
                              conv2,
                              batch_norm2,
                              activation)

# stage 8 part 1

conv1 = torch.nn.ConvTranspose2d(in_channels = y2.shape[1],
                                 out_channels = 1,
                                 kernel_size = 3,
                                 stride = 2,
                                 padding = 0)
y1 = conv1(y2)

batch_norm1 = torch.nn.BatchNorm2d(1)
y1 = batch_norm1(y1)

y1 = activation(y1)

# stage 8 part 2

zero_pad = torch.nn.ZeroPad2d(padding = (0,1,0,0))
conv2 = torch.nn.Conv2d(in_channels = y1.shape[1],
                        out_channels = y1.shape[1],
                        kernel_size = (4,3),
                        stride = 1,
                        padding = 0)
y2 = conv2(zero_pad(y1))

batch_norm2 = torch.nn.BatchNorm2d(y1.shape[1])
y2 = batch_norm2(y2)

y2 = activation(y2)

# stage 7

stages += torch.nn.Sequential(conv1,
                              batch_norm1,
                              activation,
                              zero_pad,
                              conv2,
                              batch_norm2,
                              activation)
