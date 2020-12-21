import torch
import torchaudio

from models.recon import Autoencoder
from torch_datasets.AudioDataset import AudioDataset

from collections import OrderedDict

def run_batch(mode,net,x,loss_func,optimizer,device):
    
    # move to GPU if available
    
    x = x.to(device)
    
    with torch.set_grad_enabled(mode == 'train'):
        
        # compute reconstruction of input signal
        
        x_hat = net(x)
        
        # compute reconstruction loss as negative log-likelihood
        
        recon_loss = loss_func(x_hat,x)
        
        if mode == 'train':
        
            # compute gradients of reconstruction loss with respect to
            # parameters
            
            recon_loss.backward()
            
            # update parameters using these gradients. Minimizing the negative
            # log-likelihood is equivalent to maximizing the log-likelihood
            
            optimizer.step()
            
            # zero the accumulated parameter gradients
            
            optimizer.zero_grad()
        
    return recon_loss.item()

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize reconstruction network

net = Autoencoder().to(device)

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

# initialize datasets and dataloaders
# dataset_num can be equal to 3,4,5,6,7, or 8 only

dataset_num = 4
dataset_dir = '../datasets/' + str(dataset_num if dataset_num != 8 else 1)
dataset_split_dir = '../datasets_splits/' + str(dataset_num)
sample_rate = 2000

# optimize dataloaders with GPU if available

dl_config = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

dataset = AudioDataset(dataset_dir,
                       dataset_split_dir,
                       'train',
                       sample_rate,
                       False)

dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                         batch_size = 64,
                                         shuffle = True,
                                         **dl_config)

# initialize loss function (negative log-likelihood function for
# Bernoulli distribution). This is equivalent to the KL-divergence between
# a variational likelihood and the true likelihood

loss_func = torch.nn.MSELoss(reduction = 'mean')

# initialize optimizer. Must put net parameters on GPU before this step

optimizer = torch.optim.Adam(params = net.parameters(),
                             lr = 0.0003)

# sample a batch and extract a batch of speech signals from it

batch = next(iter(dataloader))
speech_batch_size = 10
speech_batch = batch[0][torch.where(batch[1] == 2)[0]]
x = speech_batch[:speech_batch_size]

# x,sr = torchaudio.load(filepath = '../ls_1_1.wav')
# x = torchaudio.transforms.Resample(sr,2000)(x)
# x = torch.mean(x,dim=0,keepdim=True).unsqueeze(0)

# preprocessing

# zero mean and unit variance

x = torch.div((x - x.mean(dim = 2).unsqueeze(dim = -1)),
                x.std(dim = 2).unsqueeze(dim = -1))

"""
scale each signal to be between 0 and 1. This is equivalent to:

y = {x / (x.max - x.min)} - {x.min / (x.max - x.min)}
  = (1 / x.max - x.min) * (x - x.min)

"""

# x = torch.multiply(torch.div(1,x.max(dim = 2)[0].unsqueeze(dim = -1) - 
#                                 x.min(dim = 2)[0].unsqueeze(dim = -1)),
#                     x - x.min(dim = 2)[0].unsqueeze(dim = -1))

# number of times to train on the batch

num_epochs = 1000

# overfit on the batch

for epoch in range(num_epochs):
    recon_loss = run_batch('train',net,x,loss_func,optimizer,device)
    print(recon_loss)

# reconstruct the batch and plot a sample

x_hat = net(x)

import matplotlib.pyplot as plt
import sounddevice as sd

# x_hat = torch.sigmoid(x_hat)

plt.subplot(2,1,1)
plt.plot(x[2].squeeze().detach().numpy())
plt.subplot(2,1,2)
plt.plot(x_hat[2].squeeze().detach().numpy())

# sd.play(x_hat[0].squeeze().detach().numpy(),2000)