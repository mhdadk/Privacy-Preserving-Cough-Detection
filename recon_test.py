import torch

from models.recon import Autoencoder
from torch_datasets.AudioDataset import AudioDataset

from collections import OrderedDict

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize datasets and dataloaders
# dataset_num can be equal to 3,4,5,6,7, or 8 only

dataset_num = 4
dataset_dir = '../datasets/' + str(dataset_num if dataset_num != 8 else 1)
dataset_split_dir = '../datasets_splits/' + str(dataset_num)
sample_rate = 2000
dataset = AudioDataset(dataset_dir,
                       dataset_split_dir,
                       'test',
                       sample_rate)

# initialize reconstruction network and load parameters

net = Autoencoder().to(device)
num_epochs = 20
pt_filename = 'dataset'+str(dataset_num)+'_'+str(num_epochs)+'epochs.pt'
param_path = 'parameters/recon/' + pt_filename
net.load_state_dict(torch.load(param_path))

# register forward hooks to record the output tensors from each layer

layer_output = OrderedDict()

def get_output(module_name):
    def hook(module,input,output):
        layer_output[module_name] = output.detach()
    return hook

for name, module in net.named_modules():
    if 'out' in name:
        parent, child = name.split('.')
        net.__getattr__(parent).__getattr__(child).register_forward_hook(
                                    get_output(name))

# sample an audio signal

idx = torch.randint(0,len(dataset),(1,)).item()
x,label = dataset[idx]
x = torch.unsqueeze(x, dim = 0)

# preprocessing

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

# reconstruct the audio signal

x_hat = net(x)
