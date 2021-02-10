import torch
import torchaudio
import time
import csv

from models.recon3 import Autoencoder
from torch_datasets.AudioDataset import AudioDataset

def run_batch_disc(mode,x,labels,transforms,net,loss_func,device):
    
    log, mel, spec = transforms
    
    # compute logarithmic Mel-scale spectrogram of the filtered signal
    
    x = log(mel(spec(x))
    
    # with torch.set_grad_enabled(mode == 'train'):        
    
    # logits must have the same shape as labels
    
    logits = net(x).squeeze(dim = 1)
    
    # compute negative log-likelihood (NLL) using logits
    
    NLL = loss_func(logits,labels)
    
    # record predictions. since sigmoid(0) = 0.5, then negative values
    # correspond to class 0 and positive values correspond to class 1
    
    preds = logits > 0
    
    # record correct predictions
    
    true_preds = torch.sum(preds == labels)
    
    return NLL,true_preds.item()

def run_batch_recon(mode,x,spec,net,loss_func,device):
    
    # compute magnitude spectrogram of the filtered signal
    
    x = spec(x)
    x = torchaudio.functional.magphase(x)[0]
    
    # scale each magnitude spectrogram in the batch to the interval [0,1]
    
    scale_factor = x.amax(dim=(2,3))[(..., ) + (None, ) * 2]
    x_scaled = x / scale_factor
    
    # with torch.set_grad_enabled(mode == 'train'):
        
    # compute reconstruction of the scaled magnitude spectrogram
    
    x_hat = net(x_scaled)
    
    # unscale magnitude spectrogram back to normal values
    
    x_hat = x_hat * scale_factor
    
    # compute reconstruction loss
    
    recon_loss = loss_func(x_hat,x)
    
    return recon_loss

def run_batch(batch,mode,transforms,lambda_adv,filt,net,loss_funcs,
              optimizer,device):
    
    # unpack the batch and loss functions
    
    x,labels = batch
    disc_loss_func, recon_loss_func = loss_funcs
    
    # move to GPU if available
    
    x = x.to(device)
    labels = labels.to(device).type_as(x) # needed for NLL
    
    with torch.set_grad_enabled(mode == 'train'):
        
        # filter each signal
        
        x = filt(x)
        
        # compute classification (negative log-likelihood (NLL)) loss and
        # number of correct predictions
        
        NLL, true_preds = run_batch_disc(mode,x,labels,transforms,disc_net,
                                         disc_loss_func,device)
        
        # extract speech signals from x and compute reconstruction loss
        
        where_speech = torch.nonzero(labels == 0, as_tuple = True)
        recon_loss = run_batch_recon(mode,x[where_speech],transforms[2],
                                     recon_net,recon_loss_func,device)
        
        # compute adversarial loss
        
        adv_loss = NLL - lambda_adv * recon_loss
        
        if mode == 'train':
        
            # compute gradient of adversarial loss with respect to filter,
            # discriminator network, and reconstruction network parameters
            
            adv_loss.backward()
            
            # update parameters using the gradient descent update rule
            
            optimizer.step()
            
            # zero the accumulated parameter gradients
            
            optimizer.zero_grad()
    
    return NLL.item(),true_preds.item(),recon_loss.item(),adv_loss.item()

def run_epoch(mode,dataloader,transforms,filt,net,optimizer,loss_funcs,
              lambda_adv,device):
    
    if mode == 'train':
        print('Training...')
        net.train()
    else:
        print('\nValidating...')
        net.eval()
    
    # to record epoch statistics
    
    total_stats = torch.tensor([0,0,0,0])
    
    for i,batch in enumerate(dataloader):
        
        # track progress
        
        print('\rProgress: {:.2f}%'.format((i+1)/len(dataloader)*100),
              end='',flush=True)
        
        # train or validate over the batch
        
        stats = run_batch(batch,mode,transforms,lambda_adv,filt,net,
                          loss_funcs,optimizer,device)
        
        # record running statistics
        
        total_stats += torch.tensor(stats)
    
    mean_stats = total_stats / len(dataloader.dataset)
    
    return mean_stats.tolist()

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize reconstruction network

net = Autoencoder(inst_norm = False, num_channels = 32).to(device)
@torch.no_grad()
def init_params(m):
    for param in m.parameters():
        #if param.ndim >= 2: # for weights only not biases
        torch.nn.init.uniform_(param,-2,2)
net.apply(init_params)

# number of epochs to train and validate for

num_epochs = 120

# initialize datasets and dataloaders

raw_data_dir = 'data/raw'
window_length = 1.5 # seconds
sample_rate = 16000
dataloaders = {}

win_length_sec = 0.012
win_length = int(sample_rate * win_length_sec)
# need 50% overlap to satisfy constant-overlap-add constraint to allow
# for perfect reconstruction using inverse STFT
hop_length = int(sample_rate * win_length_sec / 2)

spec = torchaudio.transforms.Spectrogram(n_fft = 512,
                                         win_length = win_length,
                                         hop_length = hop_length,
                                         pad = 0,
                                         window_fn = torch.hann_window,
                                         power = None,
                                         normalized = False,
                                         wkwargs = None).to(device)

# csv file to write training and validation results

fp = open('results/train_val/recon.csv', mode = 'w')
csv_writer = csv.writer(fp,
                        delimiter = ',',
                        lineterminator = '\n')
csv_writer.writerow(['Window length (seconds)',window_length])
csv_writer.writerow(['Sample rate',sample_rate])

# where to save parameters in a .pt file

pt_filename = (str(window_length).replace('.','-') + 's' + 
               '_' + str(num_epochs) + 'epochs.pt')
param_path = 'parameters/recon/' + pt_filename

# optimize dataloaders with GPU if available

dl_config = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# batch sizes for training, validation, and testing

train_batch_size = 64
val_batch_size = 64
csv_writer.writerow(['Training batch size',train_batch_size])
csv_writer.writerow(['Validation batch size',val_batch_size])

for mode,batch_size in [('train',train_batch_size),
                        ('val',val_batch_size)]:
    
    dataset = AudioDataset(raw_data_dir,window_length,sample_rate,mode,
                           normalize = True,only_speech = True)
    
    dataloaders[mode] = torch.utils.data.DataLoader(
                               dataset = dataset,
                               batch_size = batch_size,
                               shuffle = True,
                               **dl_config)

# initialize loss functions

"""
loss function defined in "Fast Spectrogram Inversion using Multi-head
Convolutional Neural Networks". Assuming 4D tensors as input.
"""

def loss_func(x_hat,x,alpha = 1):
    
    # spectral convergence
    
    num = torch.linalg.norm(x - x_hat, ord = 'fro', dim = (2,3)).squeeze()
    den = torch.linalg.norm(x, ord = 'fro', dim = (2,3)).squeeze()
    spec_conv = torch.div(num,den)
    
    # log-scale STFT magnitude loss
    
    eps = 1e-10
    log_loss = torch.linalg.norm(torch.log(x + eps) - torch.log(x_hat + eps),
                                 ord = 1, dim = (2,3)).squeeze()
    
    return torch.sum(spec_conv + alpha * log_loss)

# loss_func = torch.nn.SmoothL1Loss(reduction = 'mean',
#                                   beta = 1.0)
# csv_writer.writerow(['Loss function',loss_func.__repr__()])
# csv_writer.writerow(['Loss function beta',loss_func.beta])

# initialize optimizer. Must put net parameters on GPU before this step

optimizer = torch.optim.Adam(params = net.parameters(),
                             lr = 0.022756034459905584,
                             eps = 1e-08,
                             weight_decay = 0,
                             amsgrad = False)
csv_writer.writerow(['Optimizer',optimizer.__module__])
csv_writer.writerow(['Optimizer parameters',str(optimizer.defaults)])

# record the best validation loss across epochs

best_val_loss = 1e10

csv_writer.writerow(['Epoch',
                     'Epoch time',
                     'Training loss',
                     'Validation loss'])

# transforms

log = torchaudio.transforms.AmplitudeToDB().to(device)

# settings for spectrograms

win_length_sec = 0.012
win_length = int(sample_rate * win_length_sec)
# need 50% overlap to satisfy constant-overlap-add constraint to allow
# for perfect reconstruction using inverse STFT
hop_length = int(sample_rate * win_length_sec / 2) # used to be 64
n_mels = 128
n_fft = 512 # used to be 1024

mel = torchaudio.transforms.MelScale(n_mels = n_mels,
                                     sample_rate = sample_rate,
                                     f_min = 0.0,
                                     f_max = None,
                                     n_stft = n_fft)

spec = torchaudio.transforms.Spectrogram(n_fft = n_fft,
                                         win_length = win_length,
                                         hop_length = hop_length,
                                         pad = 0,
                                         window_fn = torch.hann_window,
                                         power = None,
                                         normalized = False,
                                         wkwargs = None).to(device)

if __name__ == '__main__':

    # starting time
    
    start = time.time()
    
    for epoch in range(num_epochs):
        
        # show number of epochs elapsed
        
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        
        # record the epoch start time
        
        epoch_start = time.time()
        
        # train for an epoch
    
        train_loss = run_epoch('train',
                               net,
                               spec,
                               dataloaders['train'],
                               optimizer,
                               loss_func,
                               device)
        
        # show results
        
        print('\nAverage Training loss: {:.4f}'.format(train_loss))
        
        # validate for an epoch
        
        val_loss = run_epoch('val',
                             net,
                             spec,
                             dataloaders['val'],
                             optimizer,
                             loss_func,
                             device)
        
        # show results
        
        print('\nAverage Validation Loss: {:.4f}'.format(val_loss))
        
        # show epoch time
        
        epoch_end = time.time()
        
        epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end-epoch_start))
        
        # record results
        
        csv_writer.writerow([epoch+1,
                             epoch_time,
                             '{:.4f}'.format(train_loss),
                             '{:.4f}'.format(val_loss)])
        
        print('\nEpoch Elapsed Time (HH:MM:SS): ' + epoch_time)
        
        # save the weights for the best validation accuracy
        
        if val_loss < best_val_loss:
            print('Saving checkpoint...')
            best_val_loss = val_loss
            torch.save(net.state_dict(),param_path)
        
    # show training and validation time and best validation accuracy
    
    end = time.time()
    total_time = time.strftime("%H:%M:%S",time.gmtime(end-start))
    print('\nTotal Time Elapsed (HH:MM:SS): ' + total_time)
    print('Best Validation Loss: {:.2f}'.format(best_val_loss))

fp.close()