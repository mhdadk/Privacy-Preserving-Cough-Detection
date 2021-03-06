import torch
import torchaudio
import time
import csv

from models.disc import Disc
from torch_datasets.AudioDataset import AudioDataset

def run_batch(mode,net,x,labels,loss_func,optimizer,device):
    
    # move to GPU if available
    
    x = x.to(device)
    labels = labels.to(device).type_as(x) # needed for NLL
    
    with torch.set_grad_enabled(mode == 'train'):
    
        # compute log Mel spectrogram
        
        log = torchaudio.transforms.AmplitudeToDB().to(device)
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = sample_rate,
                                                        n_fft = 1024,
                                                        n_mels = 128,
                                                        hop_length = 64).to(device)
        log_mel_spec = log(mel_spec(x))
        
        # logits must have same shape as labels
        
        logits = net(log_mel_spec).squeeze(dim = 1)
        
        # compute negative log-likelihood (NLL) using logits
        
        NLL = loss_func(logits,labels)
        
        if mode == 'train':
        
            # compute gradients of NLL with respect to parameters
            
            NLL.backward()
            
            # update parameters using these gradients. Minimizing the negative
            # log-likelihood is equivalent to maximizing the log-likelihood
            
            optimizer.step()
            
            # zero the accumulated parameter gradients
            
            optimizer.zero_grad()
    
    # record predictions. since sigmoid(0) = 0.5, then negative values
    # correspond to class 0 and positive values correspond to class 1
    
    preds = logits > 0
    
    # record correct predictions
    
    true_preds = torch.sum(preds == labels)
    
    return NLL.item(),true_preds.item()

def run_epoch(mode,net,dataloader,optimizer,loss_func,device):
    
    if mode == 'train':
        print('Training...')
        net.train()
    else:
        print('\nValidating...')
        net.eval()
    
    # to compute average negative log-likelihood (NLL) per sample
    
    total_NLL = 0
    
    # to compute training accuracy per epoch
    
    total_true_preds = 0
    
    for i,(x,labels) in enumerate(dataloader):
        
        # track progress
        
        print('\rProgress: {:.2f}%'.format((i+1)/len(dataloader)*100),
              end='',flush=True)
        
        # train or validate over the batch
        
        NLL,true_preds = run_batch(mode,net,x,labels,loss_func,
                                   optimizer,device)
        
        # record running statistics
        
        total_NLL += NLL
        total_true_preds += true_preds
        
    NLL_per_sample = total_NLL / len(dataloader.dataset)
    acc = total_true_preds / len(dataloader.dataset)
    
    return NLL_per_sample,acc

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize discriminator network

FENet_param_path = 'parameters/FENet/FENet.pkl'
net = Disc(FENet_param_path).to(device)

# number of epochs to train and validate for

num_epochs = 5

# initialize datasets and dataloaders

raw_data_dir = '../data/raw'
window_length = 1.5 # seconds
sample_rate = 16000
dataloaders = {}

# csv file to write training and validation results

fp = open('results/train_val/disc.csv', mode = 'w')
csv_writer = csv.writer(fp,
                        delimiter = ',',
                        lineterminator = '\n')
csv_writer.writerow(['Window length (seconds)',window_length])
csv_writer.writerow(['Sample rate',sample_rate])

# where to save parameters in a .pt file

pt_filename = (str(window_length).replace('.','-') + 's' + 
               '_' + str(num_epochs) + 'epochs.pt')
param_path = 'parameters/disc/' + pt_filename

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
                           only_speech = False)
    
    dataloaders[mode] = torch.utils.data.DataLoader(
                               dataset = dataset,
                               batch_size = batch_size,
                               shuffle = True,
                               **dl_config)

# initialize loss function (negative log-likelihood function for
# Bernoulli distribution)

loss_func = torch.nn.BCEWithLogitsLoss(reduction = 'sum')
csv_writer.writerow(['Loss function',loss_func.__repr__()])

# initialize optimizer. Must put net parameters on GPU before this step

optimizer = torch.optim.Adam(params = net.parameters(),
                             lr = 0.0003)
csv_writer.writerow(['Optimizer',optimizer.__module__])
csv_writer.writerow(['Optimizer parameters',str(optimizer.defaults)])

# record the best validation accuracy across epochs

best_val_acc = 0

csv_writer.writerow(['Epoch',
                     'Epoch time',
                     'Training loss',
                     'Training accuracy',
                     'Validation loss',
                     'Validation accuracy'])

if __name__ == '__main__':

    # starting time
    
    start = time.time()
    
    for epoch in range(num_epochs):
        
        # show number of epochs elapsed
        
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        
        # record the epoch start time
        
        epoch_start = time.time()
        
        # train for an epoch
    
        train_loss,train_acc = run_epoch('train',
                                         net,
                                         dataloaders['train'],
                                         optimizer,
                                         loss_func,
                                         device)
        
        # show results
        
        print('\nAverage Training loss: {:.4f}'.format(train_loss))
        print('Training Accuracy: {:.2f}%'.format(train_acc*100))
        
        # validate for an epoch
        
        val_loss,val_acc = run_epoch('val',
                                     net,
                                     dataloaders['val'],
                                     optimizer,
                                     loss_func,
                                     device)
        
        # show results
        
        print('\nAverage Validation Loss: {:.4f}'.format(val_loss))
        print('Validation Accuracy: {:.2f}%'.format(val_acc*100))
        
        # show epoch time
        
        epoch_end = time.time()
        
        epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end-epoch_start))
        
        # record results
        
        csv_writer.writerow([epoch+1,
                             epoch_time,
                             '{:.4f}'.format(train_loss),
                             '{:.4f}'.format(train_acc),
                             '{:.4f}'.format(val_loss),
                             '{:.4f}'.format(val_acc)])
        
        print('\nEpoch Elapsed Time (HH:MM:SS): ' + epoch_time)
        
        # save the weights for the best validation accuracy
        
        if val_acc > best_val_acc: #and train_acc < val_acc:
            print('Saving checkpoint...')
            best_val_acc = val_acc
            torch.save(net.state_dict(),param_path)
        
    # show training and validation time and best validation accuracy
    
    end = time.time()
    total_time = time.strftime("%H:%M:%S",time.gmtime(end-start))
    print('\nTotal Time Elapsed (HH:MM:SS): ' + total_time)
    print('Best Validation Accuracy: {:.2f}%'.format(best_val_acc*100))

fp.close()