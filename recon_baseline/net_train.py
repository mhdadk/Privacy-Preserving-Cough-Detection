import torch
import torchaudio

def train(net,dataloader,loss_func,optimizer,device):
    
    # put in training mode
    
    net.train()
    
    # to compute training accuracy
    
    num_true_pred = 0
    
    # to compute epoch training loss
    
    total_loss = 0
    
    for signals,labels in dataloader:
        
        # load onto GPU
        
        signals = signals.to(device).unsqueeze(dim=1)
        labels = labels.to(device).type_as(signals) # needed for BCE loss
        
        # compute log Mel spectrogram
        
        images = torchaudio.transforms.MelSpectrogram(sample_rate = 16000,
                                             n_fft = 1024,
                                             n_mels = 256,
                                             hop_length = 63)(signals)
        images = torchaudio.transforms.AmplitudeToDB()(images)
        
        # zero the accumulated parameter gradients
        
        optimizer.zero_grad()
        
        # outputs of net for batch input
        
        outputs = net(images).squeeze()
        
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
        
        total_loss = total_loss + loss
    
    train_loss = total_loss.item() / len(dataloader.dataset)
    
    train_acc = num_true_pred.item() / len(dataloader.dataset)
    
    return net,train_loss,train_acc
