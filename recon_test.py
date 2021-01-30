import torch
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

# config

raw_data_dir = '../data/raw'
window_length = 1.5 # seconds
sample_rate = 2000
num_epochs = 60

# initialize reconstruction network

net = Autoencoder(batch_norm = True).to(device)
pt_filename = '1-5s_120epochs.pt'
param_path = 'parameters/recon/' + pt_filename
net.load_state_dict(torch.load(param_path,map_location = device))
net.eval()

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

dataset = AudioDataset(raw_data_dir,window_length,sample_rate,'test',
                       only_speech = True)

# speech_data_dir = pathlib.Path('../datasets2/4/2_LIBRISPEECH')
# window_length = 1.5 # seconds
# sample_rate = 2000
# dataset = SpeechDataset(str(speech_data_dir),window_length,sample_rate)

# optimize dataloaders with GPU if available

dl_config = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                         batch_size = 8,
                                         shuffle = False,
                                         **dl_config)

# initialize loss function

loss_func = torch.nn.SmoothL1Loss(reduction = 'mean',
                                  beta = 1.0)

# load a batch

x = next(iter(dataloader))

#%%

x_hat,recon_loss = run_batch(x,net,'test',loss_func,None,device)

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
