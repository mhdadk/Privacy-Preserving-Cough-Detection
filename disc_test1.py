import torch
import torchaudio
from sklearn.metrics import confusion_matrix

from Disc import Disc
from AudioDataset import AudioDataset

def test_batch(net,signals,device):
    
    # compute log Mel spectrogram

    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = 16000,
                                                    n_fft = 1024,
                                                    n_mels = 128,
                                                    hop_length = 64).to(device)
    to_dB = torchaudio.transforms.AmplitudeToDB().to(device)
    images = to_dB(mel_spec(signals))#.unsqueeze(dim=1) # add grayscale image channel

    with torch.no_grad():

        # outputs of net for batch input

        outputs = net(images)
    
    # record predictions. since sigmoid(0) = 0.5, then negative values
    # correspond to class 0 and positive values correspond to class 1
    
    preds = outputs > 0
    
    return preds

def compute_metrics(labels,preds):
    
    CM = confusion_matrix(labels,preds,labels=[0,1])
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
               'sensitivity':sensitivity,
               'specificity':specificity,
               'acc':accuracy,
               'bal_acc':balanced_accuracy,
               'MCC':MCC,
               'precision':PPV,
               'NPV':NPV}
    
    return metrics

def test(net,best_param_path,dataloader,device):
    
    # load best parameters

    net.load_state_dict(torch.load(best_param_path))
    
    # put net in testing mode
    
    net.eval()
    
    print('\nTesting...')
    
    # store all labels and predictions for entire testing dataset
    
    all_preds = []
    all_labels = []
    
    for i,(signals,labels) in enumerate(dataloader):
    
        # track progress
        
        print('\rProgress: {:.2f}%'.format(i*dataloader.batch_size/
                                         len(dataloader.dataset)*100),
               end='',flush=True)
        
        # move to GPU
        
        signals = signals.to(device)
        
        # record labels
        
        all_labels.extend(labels.tolist())
        
        preds = test_batch(net,signals,device)
        
        # record predictions
        
        all_preds.extend(preds.squeeze().tolist())
    
    metrics = compute_metrics(all_labels,all_preds)
    
    return metrics

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# get discriminator network

net = Disc().to(device)

# initialize dataloader

data_split_dir = 'data_split'
sample_rate = 16000
# optimize dataloaders with GPU if available
dl_config = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
disc_dataset = AudioDataset(net_type='disc',
                            data_split_dir=data_split_dir,
                            sample_rate=sample_rate,
                            mode='test')
recon_dataset = AudioDataset(net_type='recon',
                             data_split_dir=data_split_dir,
                             sample_rate=sample_rate,
                             mode='test')
dataset = torch.utils.data.ConcatDataset([disc_dataset,recon_dataset])
dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                         batch_size = 64,
                                         shuffle = True,
                                         **dl_config)

# test and show results

metrics = test(net,'parameters/10_epochs_short_cough.pt',dataloader,device)

print('Testing results:')    
print('\nConfusion Matrix:\n{}\n'.format(metrics['CM']))
print('Sensitivity/Recall: {:.3f}'.format(metrics['sensitivity']))
print('Specificity: {:.3f}'.format(metrics['specificity']))
print('Accuracy: {:.3f}'.format(metrics['acc']))
print('Balanced Accuracy: {:.3f}'.format(metrics['bal_acc']))
print('Matthews correlation coefficient: {:.3f}'.format(metrics['MCC']))
print('Precision/PPV: {:.3f}'.format(metrics['precision']))
print('NPV: {:.3f}'.format(metrics['NPV']))