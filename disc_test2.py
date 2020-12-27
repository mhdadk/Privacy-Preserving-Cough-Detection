import os
import csv

import torch
import torchaudio
from sklearn.metrics import confusion_matrix

from models.disc import Disc

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

def test_batch(x,sr,net,device):
    
    # move to GPU if available
    
    x = x.to(device)
    
    with torch.no_grad():
    
        # compute log Mel spectrogram
        
        log = torchaudio.transforms.AmplitudeToDB().to(device)
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = sr,
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

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize discriminator network and load parameters

FENet_param_path = 'parameters/FENet/FENet.pkl'
net = Disc(FENet_param_path).to(device)
net_param_path = 'parameters/disc/dataset9_20epochs.pt'
net.load_state_dict(torch.load(net_param_path,map_location=torch.device('cpu')))

# where long audio signals containing coughs are located

data_dir = '../datasets/1/1_COUGH'

# sample rate to use

new_sr = 16000

# where cough timestamps are located

cough_timestamps_path = 'utils/audio_analysis/extract_cough_windows/' + \
                        'cough_timestamps.csv'

# store the first and last cough sample numbers

cough_timestamps = {}
fp = open(cough_timestamps_path)
csv_reader = csv.reader(fp,delimiter=',')
prev_filename = ''

for row in csv_reader:
    file1,file2,_ = row[0].split('_')
    filename = file1 + '_' + file2 + '.wav'
    cough_start = round(float(row[1]) * new_sr)
    cough_end = round(float(row[2]) * new_sr)
    if prev_filename == filename:
        cough_timestamps[filename].append([cough_start,cough_end])
    else:
        cough_timestamps[filename] = [[cough_start,cough_end]]
    prev_filename = filename
    
"""
given the locations of the coughs in the files, iterate through each
file while sliding an overlapping window, recording the prediction for
each window by the classifier, recording the label for each window
using a threshold, and accumulating prediction statistics to form
a confusion matrix for all the windows for all files
"""

threshold = 0.5

# to accumulate prediction statistics and form a confusion matrix

CM = {'TP':0,
      'TN':0,
      'FP':0,
      'FN':0}

# size of sliding window
    
window_length = 1.5 # seconds
window_length = int(new_sr * window_length) # samples
overlap = 2/3 # fraction of window_length
step_size = int(window_length * (1 - overlap))

for i,(filename,cough_locs) in enumerate(cough_timestamps.items()):
    
    print('\rProgress: {:.2f}%'.format(((i+1) / len(cough_timestamps)) * 100),
          end='',flush=True)
    
    # filepath
    
    path = os.path.join(data_dir,filename)
    
    # load the audio signal
    
    x,old_sr = torchaudio.load(filepath = path)
    
    # resample to new_sr Hz
    
    x = torchaudio.transforms.Resample(old_sr,new_sr)(x)
    
    # make mono
    
    x = torch.mean(x,dim=0,keepdim=True)
    
    # slide overlapping window
    
    for j in range(0, x.shape[1] - window_length + 1, step_size):
    
        # to record the maximum intersection ratio
        
        prev_ratio = 0;
        
        # iterate over each cough to check if it intersects with the window
        
        for cough_start,cough_end in cough_locs:
            
            # compute intersection length
            
            lower = max(cough_start, j)
            upper = min(cough_end, j + window_length - 1)
            
            # if lower > upper, then there is no intersection between the
            # cough and the window, so intersection_length = 0
            
            if lower <= upper:
                intersection_length = upper - lower + 1
            else:
                intersection_length = 0
            
            # compute maximum intersection ratio
            
            ratio = max(intersection_length / (cough_end - cough_start + 1),
                        prev_ratio)
            
            # no need to check other coughs if ratio = 1
            
            if ratio == 1.0:
                break
            
            prev_ratio = ratio;
        
        # once the maximum ratio for the window is computed, compare it to a
        # threshold to determine the label for the window
        
        window_label = ratio >= threshold
        
        # classifier prediction for window
        
        pred = test_batch(x[0,j:j+window_length].unsqueeze(0).unsqueeze(0),
                          new_sr,net,device)
        
        # accumulate prediction statistics
        
        if pred == 1 and window_label == 1: # true positive
            CM['TP'] += 1
        elif pred == 1 and window_label == 0: # false positive
            CM['FP'] += 1
        elif pred == 0 and window_label == 0: # true negative
            CM['TN'] += 1
        elif pred == 0 and window_label == 1: # false negative
            CM['FN'] += 1

# compute prediction statistics

TP = CM['TP']
FP = CM['FP']
TN = CM['TN']
FN = CM['FN']
sensitivity = TP/(TP+FN) # true positive rate (TPR) or recall
specificity = TN/(TN+FP) # true negative rate (TNR)
accuracy = (TP+TN)/(TP+TN+FP+FN)
balanced_accuracy = (sensitivity+specificity)/2
MCC = (TP*TN - FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)
PPV = TP/(TP+FP) # precision
NPV = TN/(TN+FN)
