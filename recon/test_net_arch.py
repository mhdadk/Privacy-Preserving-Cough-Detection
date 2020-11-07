import torch

x = torch.randn(32,1,3000)

# initial filter

filt = torch.nn.Conv1d(in_channels = 1,
                       out_channels = 1,
                       kernel_size = 40,
                       stride = 1)

y1 = filt(x)

# reconstruction network

# encode

pool1 = torch.nn.MaxPool1d(kernel_size = 2)

y2 = pool1(y1)

conv1 = torch.nn.Conv1d(in_channels = 1,
                        out_channels = 1,
                        kernel_size = 40,
                        stride = 1)

y3 = conv1(y2)

y4 = pool1(y3)

conv2 = torch.nn.Conv1d(in_channels = 1,
                        out_channels = 1,
                        kernel_size = 40,
                        stride = 1)

y5 = conv2(y4)

y6 = pool1(y5)

# decode

us1 = torch.nn.Upsample(scale_factor = 3,
                        mode = 'nearest')

y7 = us1(y6)

conv3 = torch.nn.Conv1d(in_channels = 1,
                        out_channels = 1,
                        kernel_size = 15,
                        stride = 1,
                        padding = 0)

y8 = conv3(y7)

us2 = torch.nn.Upsample(scale_factor = 3,
                        mode = 'nearest')

y9 = us2(y8)

conv4 = torch.nn.Conv1d(in_channels = 1,
                        out_channels = 1,
                        kernel_size = 19,
                        stride = 1,
                        padding = 0)

y10 = conv4(y9)
