import torch

from Disc import Disc
from AudioDataset import AudioDataset

from net_train import train
from net_validate import validate
from net_test import test

import copy
import time

# for reproducibility

torch.manual_seed(0)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# get discriminator network

net = Disc().to(device)

# initialize datasets and dataloaders

data_split_dir = '../../data_split'
sample_rate = 16000
datasets = {}
# optimize dataloaders with GPU if available
dl_config = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
dataloaders = {}

for dataset,batch_size in [('train',64),('val',64),('test',64)]:
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
    
    epoch_start = time.time()
    
    print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 30)
    
    net,train_loss,train_acc = train(net,
                                     dataloaders['train'],
                                     loss_func,
                                     optimizer,
                                     device)
    
    print('Training Loss: {:.4f}'.format(train_loss))
    print('Training Accuracy: {:.2f}%'.format(train_acc*100))
    
    val_loss,val_acc = validate(net,
                                dataloaders['val'],
                                loss_func,
                                device)   
    
    # scheduler.step()
    
    print('Validation Loss: {:.4f}'.format(train_loss))
    print('Validation Accuracy: {:.2f}%'.format(train_acc*100))
    
    epoch_end = time.time()
    
    epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end-epoch_start))
    
    print('Epoch Elapsed Time (HH:MM:SS): ' + epoch_time)
    
    # save the weights for the best validation accuracy
        
    if val_acc > best_val_acc:
        
        print('Saving checkpoint...')
        
        best_val_acc = val_acc
        
        # deepcopy needed because a dict is a mutable object
        
        best_parameters = copy.deepcopy(net.state_dict())
        
        torch.save(net.state_dict(),
                   '../../models/mobilenet_best.pt')

end = time.time()
total_time = time.strftime("%H:%M:%S",time.gmtime(end-start))
print('\nTotal Time Elapsed (HH:MM:SS): ' + total_time)
print('Best Validation Accuracy: {:.2f}%'.format(best_val_acc*100))

print('\nTesting...')

metrics = test(net,dataloaders['test'],device)

print('\nConfusion Matrix:\n{}\n'.format(metrics['CM']))
print('Sensitivity/Recall: {:.3f}'.format(metrics['sens']))
print('Specificity: {:.3f}'.format(metrics['spec']))
print('Accuracy: {:.3f}'.format(metrics['acc']))
print('Balanced Accuracy: {:.3f}'.format(metrics['bal_acc']))
print('Matthews correlation coefficient: {:.3f}'.format(metrics['MCC']))
print('Precision/PPV: {:.3f}'.format(metrics['PPV']))
print('NPV: {:.3f}'.format(metrics['NPV']))