import torch
import torchaudio

from Disc2 import Disc
from AudioDataset2 import AudioDataset

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

dl_config = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}

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
                                mode=dataset,
                                device=device)
    recon_dataset = AudioDataset(net_type='recon',
                                 data_split_dir=data_split_dir,
                                 sample_rate=sample_rate,
                                 mode=dataset,
                                 device=device)
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
                             lr = 0.003)

# initialize learning rate scheduler
    
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer,
#                                           step_size = 3,
#                                           gamma = 0.5,
#                                           last_epoch = -1)

# whether to sample a single batch for a trial run

trial_run = False

# otherwise, set the number of epochs to train and validate for

if not trial_run:
    num_epochs = 10

# record the best validation accuracy across epochs

best_val_acc = 0

# helper functions

def train_batch(images,labels,device,net,loss_func,optimizer):
    
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
    
    for i,(images,labels) in enumerate(dataloader):
        
        # track progress
        
        print('\rProgress: {:.2f}%'.format(i*dataloader.batch_size/
                                         len(dataloader.dataset)*100),
              end='',flush=True)
        
        # move to GPU
    
        images = images.to(device)
        labels = labels.to(device).type_as(images) # needed for BCE loss
        
        # train over the batch
        
        loss,preds = train_batch(images,labels,device,net,loss_func,
                                 optimizer)
        
        # record running statistics
        
        num_true_pred += torch.sum(preds == labels).item()
        total_loss += loss.item()
    
    train_loss = total_loss / len(dataloader.dataset)
    train_acc = num_true_pred / len(dataloader.dataset)
    
    return train_loss,train_acc

def val_batch(images,labels,device,net,loss_func):
    
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
    
    for i,(images,labels) in enumerate(dataloader):
        
        # track progress
        
        print('\rProgress: {:.2f}%'.format(i*dataloader.batch_size/
                                         len(dataloader.dataset)*100),
              end='',flush=True)
        
        # move to GPU
        
        images = images.to(device)
        labels = labels.to(device).type_as(images) # needed for BCE loss
        
        # validate over the batch
        
        loss,preds = val_batch(images,labels,device,net,loss_func)
        
        # record running statistics
        
        num_true_pred += torch.sum(preds == labels).item()
        total_loss += loss.item()
    
    val_loss = total_loss / len(dataloader.dataset)
    val_acc = num_true_pred / len(dataloader.dataset)
    
    return val_loss,val_acc

if __name__ == '__main__':

    # starting time
    
    start = time.time()
    
    if trial_run:
    
        # record the epoch start time
        
        epoch_start = time.time()
        
        # training ###########################################################
        
        # put net in training mode
        
        net.train()
        
        # sample a batch
        
        loader = dataloaders['train']
        signals,labels = next(iter(loader))
        
        # move to GPU
            
        signals = signals.to(device)
        labels = labels.to(device).type_as(signals) # needed for BCE loss
        
        # train over the batch
        
        loss,preds = train_batch(signals,labels,device,net,loss_func,
                                 optimizer)
        
        # compute training accuracy over batch
        
        train_acc = torch.sum(preds == labels).item() / loader.batch_size
        
        # show results
        
        print('Training Loss: {:.4f}'.format(loss))
        print('Training Accuracy: {:.2f}%'.format(train_acc*100))
        
        # validation #########################################################
        
        # put net in testing mode
                      
        net.eval()
        
        # sample a batch
        
        loader = dataloaders['val']
        signals,labels = next(iter(loader))
        
        # move to GPU
        
        signals = signals.to(device)
        labels = labels.to(device).type_as(signals) # needed for BCE loss
        
        # validate over the batch
        
        loss,preds = val_batch(signals,labels,device,net,loss_func)
        
        # compute training accuracy over batch
        
        val_acc = torch.sum(preds == labels).item() / loader.batch_size
        
        # show results
        
        print('Validation Loss: {:.4f}'.format(loss))
        print('Validation Accuracy: {:.2f}%'.format(val_acc*100))
        
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
        
        # testing ############################################################
        
        # load best parameters
        
        net.load_state_dict(torch.load('best_param.pt'))
        
        # put net in testing mode
        
        net.eval()
        
        # sample a batch
        
        loader = dataloaders['test']
        signals,labels = next(iter(loader))
        
        # move to GPU
        
        signals = signals.to(device)
        
        # test the batch
        
        preds = test_batch(net,signals,device).squeeze().tolist()
        
        # compute metrics
        
        metrics = compute_metrics(labels,preds)
        
        print('\nConfusion Matrix:\n{}\n'.format(metrics['CM']))
        print('Sensitivity/Recall: {:.3f}'.format(metrics['sensitivity']))
        print('Specificity: {:.3f}'.format(metrics['specificity']))
        print('Accuracy: {:.3f}'.format(metrics['acc']))
        print('Balanced Accuracy: {:.3f}'.format(metrics['bal_acc']))
        print('Matthews correlation coefficient: {:.3f}'.format(metrics['MCC']))
        print('Precision/PPV: {:.3f}'.format(metrics['precision']))
        print('NPV: {:.3f}'.format(metrics['NPV']))
    
    else:
    
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
