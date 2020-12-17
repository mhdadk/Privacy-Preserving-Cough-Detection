import torch
import time

from models.recon import Autoencoder
from torch_datasets.AudioDataset import AudioDataset

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

def run_epoch(mode,net,dataloader,optimizer,loss_func,device):
    
    if mode == 'train':
        print('Training...')
        net.train()
    else:
        print('\nValidating...')
        net.eval()
    
    # to compute average negative log-likelihood per sample
    
    total_NLL = 0
    
    for i,(x,_) in enumerate(dataloader):
        
        # track progress
        
        print('\rProgress: {:.2f}%'.format(i*dataloader.batch_size/
                                         len(dataloader.dataset)*100),
              end='',flush=True)
        
        # train or validate over the batch
        
        NLL = run_batch(mode,net,x,loss_func,optimizer,device)
        
        # record running statistics
        
        total_NLL += NLL
        
    NLL_per_sample = total_NLL / len(dataloader.dataset)
    
    return NLL_per_sample

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize reconstruction network

net = Autoencoder().to(device)

# number of epochs to train and validate for

num_epochs = 20

# initialize datasets and dataloaders
# dataset_num can be equal to 3,4,5,6,7, or 8 only

dataset_num = 4
dataset_dir = '../datasets/' + str(dataset_num if dataset_num != 8 else 1)
dataset_split_dir = '../datasets_splits/' + str(dataset_num)
sample_rate = 2000
dataloaders = {}

# where to save parameters in a .pt file

pt_filename = 'dataset'+str(dataset_num)+'_'+str(num_epochs)+'epochs.pt'
param_path = 'parameters/recon/' + pt_filename

# optimize dataloaders with GPU if available

dl_config = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

# batch sizes for training, validation, and testing

train_batch_size = 16
val_batch_size = 16

for mode,batch_size in [('train',train_batch_size),
                        ('val',val_batch_size)]:
    
    dataset = AudioDataset(dataset_dir,
                           dataset_split_dir,
                           mode,
                           sample_rate)
    
    dataloaders[mode] = torch.utils.data.DataLoader(
                               dataset = dataset,
                               batch_size = batch_size,
                               shuffle = True,
                               **dl_config)

# initialize loss function (negative log-likelihood function for
# Bernoulli distribution). This is equivalent to the KL-divergence between
# a variational likelihood and the true likelihood

loss_func = torch.nn.BCELoss(reduction = 'sum')

# initialize optimizer. Must put net parameters on GPU before this step

optimizer = torch.optim.Adam(params = net.parameters(),
                             lr = 0.0003)

# record the best validation loss across epochs

best_val_loss = 1e10

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
                               dataloaders['train'],
                               optimizer,
                               loss_func,
                               device)
        
        # show results
        
        print('\nAverage Training loss: {:.4f}'.format(train_loss))
        
        # validate for an epoch
        
        val_loss = run_epoch('val',
                             net,
                             dataloaders['val'],
                             optimizer,
                             loss_func,
                             device)
        
        # show results
        
        print('\nAverage Validation Loss: {:.4f}'.format(val_loss))
        
        # show epoch time
        
        epoch_end = time.time()
        
        epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end-epoch_start))
        
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
    print('Best Validation Loss: {:.2f}%'.format(best_val_loss))
