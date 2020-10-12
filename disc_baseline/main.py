import torch
import torchaudio
from sklearn.metrics import confusion_matrix

from Disc import Disc
from AudioDataset import AudioDataset

import copy
import time

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# get discriminator network

net = Disc().to(device)

# initialize datasets and dataloaders

data_split_dir = 'data_split'
sample_rate = 16000
datasets = {}
dataloaders = {}

# optimize dataloaders with GPU if available

dl_config = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

# batch sizes for training, validation, and testing

train_batch_size = 64
val_batch_size = 64
test_batch_size = 64

for dataset,batch_size in [('train',train_batch_size),
                           ('val',val_batch_size),
                           ('test',test_batch_size)]:
    
    disc_dataset = AudioDataset(net_type='disc',
                                data_split_dir=data_split_dir,
                                sample_rate=sample_rate,
                                mode=dataset)
    recon_dataset = AudioDataset(net_type='disc',
                                 data_split_dir=data_split_dir,
                                 sample_rate=sample_rate,
                                 mode=dataset)
    datasets[dataset] = torch.utils.data.ConcatDataset([disc_dataset,
                                                        recon_dataset])
    dataloaders[dataset] = torch.utils.data.DataLoader(
                               dataset = datasets[dataset],
                               batch_size = batch_size,
                               shuffle = True,
                               **dl_config)

# initialize loss function

loss_func = torch.nn.BCEWithLogitsLoss(reduction='sum')

# initialize optimizer. Must put net parameters on GPU before this step

optimizer = torch.optim.Adam(params = net.parameters(),
                             lr = 0.0003)

# initialize learning rate scheduler
    
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer,
#                                           step_size = 3,
#                                           gamma = 0.5,
#                                           last_epoch = -1)

# number of epochs to train and validate for

num_epochs = 20

# best validation accuracy

best_val_acc = 0

# starting time

start = time.time()

for epoch in range(num_epochs):
    
    # record the epoch start time
    
    epoch_start = time.time()
    
    # training #####################################################################
    
    # put net in training mode
    
    net.train()
    print('Training...')
    
    # record the number of correct predictions to compute
    # training accuracy over entire epoch
    
    num_true_pred = 0
    
    # to compute total training loss over entire epoch
    
    total_loss = 0
    
    # show number of epochs elapsed
    
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    
    for i,(signals,labels) in enumerate(dataloaders['train']):
        
        # track progress
        
        print('Progress: {:.2f}%'.format(i*dataloaders['train'].batch_size/len(dataloaders['train'])),
              end='\r',flush=True)
        
        # move to GPU
        
        signals = signals.to(device)
        labels = labels.to(device).type_as(signals) # needed for BCE loss
        
        # compute log Mel spectrogram
        
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = 16000,
                                                        n_fft = 1024,
                                                        n_mels = 256,
                                                        hop_length = 63).to(device)
        to_dB = torchaudio.transforms.AmplitudeToDB().to(device)
        images = to_dB(mel_spec(signals)).unsqueeze(dim=1) # add grayscale image channel
        
        # zero the accumulated parameter gradients
        
        optimizer.zero_grad()
        
        # outputs of net for batch input
        
        outputs = net(images).squeeze() # needed for loss_func
        
        # compute (mean) loss
        
        loss = loss_func(outputs,labels)
        
        # compute loss gradients with respect to parameters
        
        loss.backward()
        
        # update parameters according to optimizer
        
        optimizer.step()
        
        # record running statistics
        
        # since sigmoid(0) = 0.5, then negative values correspond to class 0
        # and positive values correspond to class 1
        
        class_preds = outputs > 0
        num_true_pred = num_true_pred + torch.sum(class_preds == labels)
        
        # loss is not mean-reduced
        
        total_loss += loss
    
    train_loss = total_loss.item() / len(dataloaders['train'])
    
    train_acc = num_true_pred.item() / len(dataloaders['train'])
    
    print('Training Loss: {:.4f}'.format(train_loss))
    print('Training Accuracy: {:.2f}%'.format(train_acc*100))
    
    # validation #########################################################
    
    # put net in testing mode
              
    net.eval()
    print('\nValidating...\n')
    
    # record the number of correct predictions to compute
    # validation accuracy over entire epoch
    
    num_true_pred = 0
    
    # to compute total validation loss over entire epoch
    
    total_loss = 0
    
    # show number of epochs elapsed
    
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    
    for i,(signals,labels) in enumerate(dataloaders['val']):
        
        # track progress
        
        print('Progress: {:.2f}%'.format(i*dataloaders['val'].batch_size/len(dataloaders['val'])),
              end='\r',flush=True)
        
        # move to GPU
        
        signals = signals.to(device)
        labels = labels.to(device).type_as(signals) # needed for BCE loss
        
        # compute log Mel spectrogram
        
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = 16000,
                                                        n_fft = 1024,
                                                        n_mels = 256,
                                                        hop_length = 63).to(device)
        to_dB = torchaudio.transforms.AmplitudeToDB().to(device)
        images = to_dB(mel_spec(signals)).unsqueeze(dim=1) # add grayscale image channel
        
        with torch.no_grad():
            
            # outputs of net for batch input

            outputs = net(images).squeeze()

            # compute loss

            loss = loss_func(outputs,labels)
        
        # record running statistics
        
        # since sigmoid(0) = 0.5, then negative values correspond to class 0
        # and positive values correspond to class 1
        
        class_preds = outputs > 0
        num_true_pred = num_true_pred + torch.sum(class_preds == labels)
        
        # loss is not mean-reduced
        
        total_loss += loss
    
    val_loss = total_loss.item() / len(dataloaders['val'])
    
    val_acc = num_true_pred.item() / len(dataloaders['val'])
    
    print('Validation Loss: {:.4f}'.format(val_loss))
    print('Validation Accuracy: {:.2f}%'.format(val_acc*100)) 
    
    # scheduler.step()
    
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

end = time.time()
total_time = time.strftime("%H:%M:%S",time.gmtime(end-start))
print('\nTotal Time Elapsed (HH:MM:SS): ' + total_time)
print('Best Validation Accuracy: {:.2f}%'.format(best_val_acc*100))

# testing ################################################################

# load best parameters

net.load_state_dict(torch.load('best_param.pt'))

# put net in testing mode

net.eval()

print('\nTesting...\n')

# store class predictions and the true labels

class_preds = []
true_labels = []

for i,(signals,labels) in enumerate(dataloaders['test']):

    # track progress
    
    print('Progress: {:.2f}%'.format(i*dataloaders['test'].batch_size/len(dataloaders['test'])),
           end='\r',flush=True)

    # move to GPU

    signals = signals.to(device)
    
    # store labels
    
    true_labels.extend(labels.tolist())

    # compute log Mel spectrogram

    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = 16000,
                                                    n_fft = 1024,
                                                    n_mels = 256,
                                                    hop_length = 63).to(device)
    to_dB = torchaudio.transforms.AmplitudeToDB().to(device)
    images = to_dB(mel_spec(signals)).unsqueeze(dim=1) # add grayscale image channel

    with torch.no_grad():

        # outputs of net for batch input

        outputs = net(images)

    # record running statistics

    # since sigmoid(0) = 0.5, then negative values correspond to class 0
    # and positive values correspond to class 1

    class_preds.extend((outputs > 0).squeeze().tolist())

CM = confusion_matrix(true_labels,class_preds,labels=[0,1])
    
TP = CM[1,1]

TN = CM[0,0]

FP = CM[0,1]

FN = CM[1,0]

sensitivity = TP/(TP+FN) # true positive rate (TPR)

specificity = TN/(TN+FP) # true negative rate (TNR)

accuracy = (TP+TN)/(TP+TN+FP+FN)

balanced_accuracy = (sensitivity+specificity)/2

# Matthews correlation coefficient

MCC = (TP*TN - FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)

# positive predictive value (or precision)

PPV = TP/(TP+FP)

# negative predictive value

NPV = TN/(TN+FN)

metrics = {'CM':CM,
           'sens':sensitivity,
           'spec':specificity,
           'acc':accuracy,
           'bal_acc':balanced_accuracy,
           'MCC':MCC,
           'PPV':PPV,
           'NPV':NPV}

print('\nConfusion Matrix:\n{}\n'.format(CM))
print('Sensitivity/Recall: {:.3f}'.format(sensitivity))
print('Specificity: {:.3f}'.format(specificity))
print('Accuracy: {:.3f}'.format(accuracy))
print('Balanced Accuracy: {:.3f}'.format(balanced_accuracy))
print('Matthews correlation coefficient: {:.3f}'.format(MCC))
print('Precision/PPV: {:.3f}'.format(PPV))
print('NPV: {:.3f}'.format(NPV))
