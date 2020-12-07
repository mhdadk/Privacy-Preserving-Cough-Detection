import torch
import torchaudio
import time

from models.disc import Disc
from torch_datasets.AudioDataset import AudioDataset

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize discriminator network

FENet_param_path = 'parameters/FENet/FENet.pkl'
net = Disc(FENet_param_path).to(device)

# initialize datasets and dataloaders

dataset_dir = '../datasets/3'
dataset_split_dir = '../datasets_splits/3'
sample_rate = 16000
dataloaders = {}

# optimize dataloaders with GPU if available

dl_config = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

# batch sizes for training, validation, and testing

train_batch_size = 64
val_batch_size = 64

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
# Bernoulli distribution)

loss_func = torch.nn.BCEWithLogitsLoss(reduction='sum')

# initialize optimizer. Must put net parameters on GPU before this step

optimizer = torch.optim.Adam(params = net.parameters(),
                             lr = 0.0003)

# number of epochs to train and validate for

num_epochs = 1

# record the best validation accuracy across epochs

best_val_acc = 0

# helper functions

def train_batch(signals,labels,device,net,loss_func,optimizer):
       
    # compute log Mel spectrogram
    
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = 16000,
                                                    n_fft = 1024,
                                                    n_mels = 128,
                                                    hop_length = 64).to(device)
    to_dB = torchaudio.transforms.AmplitudeToDB().to(device)
    images = to_dB(mel_spec(signals))#.unsqueeze(dim=1) # add grayscale image channel
    
    # zero the accumulated parameter gradients
    
    optimizer.zero_grad()
    
    # outputs of net for batch input
    
    outputs = net(images).squeeze() # needed for loss_func
    
    # compute loss
    
    loss = loss_func(outputs,labels)
    
    # compute loss gradients with respect to parameters
    
    loss.backward()
    
    # update parameters according to optimizer
    
    optimizer.step()
    
    # record predictions. since sigmoid(0) = 0.5, then negative values
    # correspond to class 0 and positive values correspond to class 1
    
    preds = outputs > 0
    
    return loss,preds

def train_epoch(net,dataloader,device,loss_func,optimizer):
    
    # put net in training mode
    
    net.train()
    print('Training...')
    
    # record the number of correct predictions to compute
    # training accuracy over entire epoch
    
    num_true_pred = 0
    
    # to compute total training loss over entire epoch
    
    total_loss = 0
    
    for i,(signals,labels) in enumerate(dataloader):
        
        # track progress
        
        print('\rProgress: {:.2f}%'.format(i*dataloader.batch_size/
                                         len(dataloader.dataset)*100),
              end='',flush=True)
        
        # move to GPU
    
        signals = signals.to(device)
        labels = labels.to(device).type_as(signals) # needed for BCE loss
        
        # train over the batch
        
        loss,preds = train_batch(signals,labels,device,net,loss_func,
                                 optimizer)
        
        # record running statistics
        
        num_true_pred += torch.sum(preds == labels).item()
        total_loss += loss.item()
    
    train_loss = total_loss / len(dataloader.dataset)
    train_acc = num_true_pred / len(dataloader.dataset)
    
    return train_loss,train_acc

def val_batch(signals,labels,device,net,loss_func):
    
    # compute log Mel spectrogram
    
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = 16000,
                                                    n_fft = 1024,
                                                    n_mels = 128,
                                                    hop_length = 64).to(device)
    to_dB = torchaudio.transforms.AmplitudeToDB().to(device)
    images = to_dB(mel_spec(signals))#.unsqueeze(dim=1) # add grayscale image channel
    
    with torch.no_grad():
        
        # outputs of net for batch input

        outputs = net(images).squeeze()

        # compute loss

        loss = loss_func(outputs,labels)
    
    # record predictions. since sigmoid(0) = 0.5, then negative values
    # correspond to class 0 and positive values correspond to class 1
    
    preds = outputs > 0
    
    return loss,preds


def val_epoch(net,dataloader,device,loss_func):
    
    # put net in testing mode
                  
    net.eval()
    print('\nValidating...')
    
    # record the number of correct predictions to compute
    # validation accuracy over entire epoch
    
    num_true_pred = 0
    
    # to compute total validation loss over entire epoch
    
    total_loss = 0
    
    for i,(signals,labels) in enumerate(dataloader):
        
        # track progress
        
        print('\rProgress: {:.2f}%'.format(i*dataloader.batch_size/
                                         len(dataloader.dataset)*100),
              end='',flush=True)
        
        # move to GPU
        
        signals = signals.to(device)
        labels = labels.to(device).type_as(signals) # needed for BCE loss
        
        # validate over the batch
        
        loss,preds = val_batch(signals,labels,device,net,loss_func)
        
        # record running statistics
        
        num_true_pred += torch.sum(preds == labels).item()
        total_loss += loss.item()
    
    val_loss = total_loss / len(dataloader.dataset)
    val_acc = num_true_pred / len(dataloader.dataset)
    
    return val_loss,val_acc

if __name__ == '__main__':

    # starting time
    
    start = time.time()
    
    for epoch in range(num_epochs):
        
        # show number of epochs elapsed
        
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        
        # record the epoch start time
        
        epoch_start = time.time()
        
        # train for an epoch
    
        train_loss,train_acc = train_epoch(net,dataloaders['train'],
                                           device,loss_func,optimizer)
        
        # show results
        
        print('\nTraining Loss: {:.4f}'.format(train_loss))
        print('Training Accuracy: {:.2f}%'.format(train_acc*100))
        
        # validate for an epoch
        
        val_loss,val_acc = val_epoch(net,dataloaders['val'],device,
                                     loss_func)
        
        # show results
        
        print('\nValidation Loss: {:.4f}'.format(val_loss))
        print('Validation Accuracy: {:.2f}%'.format(val_acc*100)) 
        
        # update learning rate
        
        # scheduler.step()
        
        # show epoch time
        
        epoch_end = time.time()
        
        epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end-epoch_start))
        
        print('\nEpoch Elapsed Time (HH:MM:SS): ' + epoch_time)
        
        # save the weights for the best validation accuracy
        
        if val_acc > best_val_acc:
            
            print('Saving checkpoint...')
            
            best_val_acc = val_acc
            
            # deepcopy needed because a dict is a mutable object
            
            best_parameters = copy.deepcopy(net.state_dict())
            
            torch.save(net.state_dict(),
                       'best_param.pt')
        
    # show training and validation time and best validation accuracy
    
    end = time.time()
    total_time = time.strftime("%H:%M:%S",time.gmtime(end-start))
    print('\nTotal Time Elapsed (HH:MM:SS): ' + total_time)
    print('Best Validation Accuracy: {:.2f}%'.format(best_val_acc*100))
