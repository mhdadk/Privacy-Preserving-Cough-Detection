import torch
import torchaudio
import pathlib
# from collections import OrderedDict

from models.recon import Autoencoder
# from torch_datasets.SpeechDataset import SpeechDataset
from torch_datasets.AudioDataset import AudioDataset

import matplotlib.pyplot as plt
import sounddevice as sd

def run_batch(x,net,mode,loss_func,optimizer,device):
    
    # move to GPU if available
    
    x = x.to(device)
    
    with torch.set_grad_enabled(mode == 'train'):
        
        # zero mean and unit variance
        
        x = torch.div((x - x.mean(dim = 2).unsqueeze(dim = -1)),
                       x.std(dim = 2).unsqueeze(dim = -1))
        
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
    
    if mode == 'train':
        return recon_loss.item()
    else:
        return x_hat,recon_loss.item()

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize reconstruction network

net = Autoencoder(batch_norm = True).to(device)

# register forward hooks to record the output tensors from each layer

# layer_output = OrderedDict()

# def get_output(module_name):
#     def hook(module,input,output):
#         layer_output[module_name] = output.detach()
#     return hook

# for name, module in net.named_modules():
#     if 'out' in name:
#         parent, child = name.split('.')
#         net.__getattr__(parent).__getattr__(child).register_forward_hook(
#                                     get_output(name))

# whether to use a single speech file or sample a batch of speech files

# single_file = False

# initialize datasets and dataloaders

raw_data_dir = '../data/raw'
window_length = 1.5 # seconds
sample_rate = 2000
dataset = AudioDataset(raw_data_dir,window_length,sample_rate,'train',
                       only_speech = True)

# speech_data_dir = pathlib.Path('../datasets2/4/2_LIBRISPEECH')
# window_length = 1.5 # seconds
# sample_rate = 2000
# dataset = SpeechDataset(str(speech_data_dir),window_length,sample_rate)

# optimize dataloaders with GPU if available

dl_config = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                         batch_size = 64,
                                         shuffle = False,
                                         **dl_config)

# initialize loss function

loss_func = torch.nn.SmoothL1Loss(reduction = 'mean',
                                  beta = 1.0)

# initialize optimizer. Must put net parameters on GPU before this step

optimizer = torch.optim.Adam(params = net.parameters(),
                             lr = 0.0003)

# load a batch

x = next(iter(dataloader))

# if single_file:
#     filepath = speech_data_dir / 'ls_1_1.wav'
#     x,sr = torchaudio.load(filepath = str(filepath))
#     x = torch.mean(x,dim=0,keepdim=True).unsqueeze(0)
#     x = torchaudio.transforms.Resample(sr,sample_rate)(x)
# else:
#     x = next(iter(dataloader))

# number of times to train on x

num_epochs = 1000

# overfit on the file/batch

net.train()

for epoch in range(num_epochs):
    recon_loss = run_batch(x,net,'train',loss_func,optimizer,device)
    print('\rReconstruction loss: {:.6f}'.format(recon_loss),end='',flush=True)

print('\nDone training')

#%% reconstruct the file/batch and plot and play a sample

net.eval()

x_hat,recon_loss = run_batch(x,net,'test',loss_func,optimizer,device)

print('Reconstruction loss: {}'.format(recon_loss))

idx = 0

plt.subplot(2,1,1)
plt.plot(x[idx].squeeze().detach().numpy())
plt.subplot(2,1,2)
plt.plot(x_hat[idx].squeeze().detach().numpy())

#%%

sd.play(x[idx].squeeze().detach().numpy(),2000)

#%%

sd.play(x_hat[idx].squeeze().detach().numpy(),2000)