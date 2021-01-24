import csv

import torch
import torchaudio
from sklearn.metrics import confusion_matrix

from models.disc import Disc
from torch_datasets.AudioDataset import AudioDataset

def save_metrics(labels,preds,metrics_path):
    
    fp = open(metrics_path,mode='w')
    csv_writer = csv.writer(fp,delimiter=',',lineterminator='\n')
    
    CM = confusion_matrix(labels,preds,labels=[0,1])
    TP = CM[1,1]
    TN = CM[0,0]
    FP = CM[0,1]
    FN = CM[1,0]
    sensitivity = TP/(TP+FN) # true positive rate (TPR)
    csv_writer.writerow(['Sensitivity/Recall','{:.3f}'.format(sensitivity)])
    specificity = TN/(TN+FP) # true negative rate (TNR)
    csv_writer.writerow(['Specificity','{:.3f}'.format(specificity)])
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    csv_writer.writerow(['Accuracy','{:.3f}'.format(accuracy)])
    balanced_accuracy = (sensitivity+specificity)/2
    csv_writer.writerow(['Balanced accuracy','{:.3f}'.format(balanced_accuracy)])
    
    # Matthews correlation coefficient
    
    MCC = (TP*TN - FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)
    csv_writer.writerow(['Matthews correlation coefficient','{:.3f}'.format(MCC)])
        
    # positive predictive value (or precision)
    
    PPV = TP/(TP+FP)
    csv_writer.writerow(['Precision/PPV','{:.3f}'.format(PPV)])
    
    # negative predictive value
    
    NPV = TN/(TN+FN)
    csv_writer.writerow(['NPV','{:.3f}'.format(NPV)])
    
    # close csv file after writing
    
    fp.close()
    
    metrics = {'CM':CM,
               'sensitivity':sensitivity,
               'specificity':specificity,
               'acc':accuracy,
               'bal_acc':balanced_accuracy,
               'MCC':MCC,
               'precision':PPV,
               'NPV':NPV}
    
    return metrics

def test_batch(net,x,device):
    
    # move to GPU if available
    
    x = x.to(device)
    
    with torch.no_grad():
    
        # compute log Mel spectrogram
        
        log = torchaudio.transforms.AmplitudeToDB().to(device)
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = sample_rate,
                                                        n_fft = 1024,
                                                        n_mels = 128,
                                                        hop_length = 64).to(device)
        log_mel_spec = log(mel_spec(x))
        
        # logits must have same shape as labels
        
        logits = net(log_mel_spec).squeeze(dim = 1)
    
        # record predictions. since sigmoid(0) = 0.5, then negative values
        # correspond to class 0 and positive values correspond to class 1
        
        preds = logits > 0
    
    return preds

def test_epoch(net,dataloader,device):
    
    print('Testing...')
    net.eval()
    
    # to store predictions and labels
    
    all_labels = []
    all_preds = []
    
    for i,(x,labels) in enumerate(dataloader):
        
        # track progress
        
        print('\rProgress: {:.2f}%'.format(i*dataloader.batch_size/
                                         len(dataloader.dataset)*100),
              end='',flush=True)
        
        # store labels per batch
        
        all_labels.extend(labels.tolist())
        
        # get predictions        
        
        preds = test_batch(net,x,device)
        
        # store predictions per batch
        
        all_preds.extend(preds.tolist())
    
    return all_labels,all_preds

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize discriminator network and load parameters

FENet_param_path = 'parameters/FENet/FENet.pkl'
net = Disc(FENet_param_path).to(device)
window_length = 1.0 # seconds
net_param_path = 'parameters/disc/{}_5epochs.pt'.format(str(window_length).replace('.','-')+'s')
net.load_state_dict(torch.load(net_param_path,map_location=device))

# initialize dataloader

raw_data_dir = '../data/raw'
sample_rate = 16000

# optimize dataloaders with GPU if available

dl_config = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

dataset = AudioDataset(raw_data_dir,window_length,sample_rate,'test')

batch_size = 64

dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                         batch_size = batch_size,
                                         shuffle = True,
                                         **dl_config)

# get labels and predictions

labels,preds = test_epoch(net,dataloader,device)

# compute performance metrics and save them to a .csv file

metrics_path = 'test_results/disc/{}_5epochs.csv'.format(str(window_length).replace('.','-')+'s')
metrics = save_metrics(labels,preds,metrics_path)

print('\nTesting results:')    
print('\nConfusion Matrix:\n{}\n'.format(metrics['CM']))
print('Sensitivity/Recall: {:.3f}'.format(metrics['sensitivity']))
print('Specificity: {:.3f}'.format(metrics['specificity']))
print('Accuracy: {:.3f}'.format(metrics['acc']))
print('Balanced Accuracy: {:.3f}'.format(metrics['bal_acc']))
print('Matthews correlation coefficient: {:.3f}'.format(metrics['MCC']))
print('Precision/PPV: {:.3f}'.format(metrics['precision']))
print('NPV: {:.3f}'.format(metrics['NPV']))