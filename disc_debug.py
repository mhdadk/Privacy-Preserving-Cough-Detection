import csv
import matplotlib.pyplot as plt

import torch
import torchaudio

from models.disc import Disc

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

def save_metrics(CM,metrics_path,):
    
    fp = open(metrics_path,mode='w')
    csv_writer = csv.writer(fp,delimiter=',',lineterminator='\n')
    csv_writer.writerow(['1_COUGH/fsd_14.wav contains coughs and the network was trained on']) # header
    
    TP = CM['TP']
    FP = CM['FP']
    TN = CM['TN']
    FN = CM['FN']
    sensitivity = TP/(TP+FN+1e-10) # true positive rate (TPR)
    csv_writer.writerow(['Sensitivity/Recall','{:.3f}'.format(sensitivity)])
    specificity = TN/(TN+FP+1e-10) # true negative rate (TNR)
    csv_writer.writerow(['Specificity','{:.3f}'.format(specificity)])
    accuracy = (TP+TN)/(TP+TN+FP+FN+1e-10)
    csv_writer.writerow(['Accuracy','{:.3f}'.format(accuracy)])
    balanced_accuracy = (sensitivity+specificity)/2
    csv_writer.writerow(['Balanced accuracy','{:.3f}'.format(balanced_accuracy)])
    
    # Matthews correlation coefficient
    
    MCC = (TP*TN - FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5+1e-10)
    csv_writer.writerow(['Matthews correlation coefficient','{:.3f}'.format(MCC)])
        
    # positive predictive value (or precision)
    
    PPV = TP/(TP+FP+1e-10)
    csv_writer.writerow(['Precision/PPV','{:.3f}'.format(PPV)])
    
    # negative predictive value
    
    NPV = TN/(TN+FN+1e-10)
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

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# length of the window to be used to evaluate in seconds

window_length1 = 1.0

# initialize discriminator network and load parameters

FENet_param_path = 'parameters/FENet/FENet.pkl'
net = Disc(FENet_param_path).to(device)
net_param_path = 'parameters/disc/{}_5epochs.pt'.format(str(window_length1).replace('.','-')+'s')
net.load_state_dict(torch.load(net_param_path,map_location=device))
net.eval()

# file that contains coughs and the network was trained on (00)

file1 = '../data/raw/1_COUGH/esc_29.wav'

# location of coughs in seconds

cough_locs1 = [[0.5022448979591837,0.8385487528344672],
               [0.9304988662131519,1.3811337868480726],
               [1.730952380952381,2.2174603174603176],
               [2.8497732426303855,3.1950340136054423],
               [3.9372108843537417,4.251088435374149],
               [4.280272108843538,4.5941496598639455]]

# file that contains coughs and the network was tested on (01)

file2 = '../data/raw/1_COUGH/fsd_14.wav'

# location of coughs in seconds

cough_locs2 = [[1.1493197278911564,1.551859410430839],
               [1.6882539682539683,2.0063945578231293],
               [2.9024943310657596,3.292063492063492],
               [4.337505668934241,4.675124716553288],
               [6.3212018140589565,6.658820861678005]]

# file that contains no coughs and the network was trained on (10)

file3 = '../data/raw/0_FSDKAGGLE2018/fsd_2104.wav'

# file that contains no coughs and the network was tested on (11)

file4 = '../data/raw/0_FSDKAGGLE2018/fsd_2559.wav'

# sample rate to use

new_sr = 16000

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

# load the audio signal

x,old_sr = torchaudio.load(filepath = file2)

# resample to new_sr Hz

x = torchaudio.transforms.Resample(old_sr,new_sr)(x)

# make mono

x = torch.mean(x,dim=0,keepdim=True)

# plot signal

# fig = plt.figure()
# plt.plot(x.numpy()[0],figure=fig)

# size of sliding window

window_length = int(new_sr * window_length1) # samples
overlap = 2/3 # fraction of window_length
step_size = int(window_length * (1 - overlap))

# slide overlapping window

for j in range(0, x.shape[1] - window_length + 1, step_size):
    
    # plot window
    
    # fig = plt.figure()
    # plt.plot(x.numpy()[0,j:j+window_length],figure=fig)
    
    # to record the maximum intersection ratio
    
    prev_ratio = 0;
    
    # iterate over each cough to check if it intersects with the window
    
    for cough_start,cough_end in cough_locs2:
        
        # convert to sample number
        
        cough_start = round(cough_start * new_sr)
        cough_end = round(cough_end * new_sr)
        
        # plot cough
        
        # fig = plt.figure()
        # plt.plot(x.numpy()[0,cough_start:cough_end],figure = fig)
        
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

# compute performance metrics and save them to a .csv file

metrics_path = 'test_results/disc_debug/{}_5epochs_01_2.csv'.format(str(window_length1).replace('.','-')+'s')
metrics = save_metrics(CM,metrics_path)

print('\nTesting results:')    
print('\nConfusion Matrix:\n{}\n'.format(metrics['CM']))
print('Sensitivity/Recall: {:.3f}'.format(metrics['sensitivity']))
print('Specificity: {:.3f}'.format(metrics['specificity']))
print('Accuracy: {:.3f}'.format(metrics['acc']))
print('Balanced Accuracy: {:.3f}'.format(metrics['bal_acc']))
print('Matthews correlation coefficient: {:.3f}'.format(metrics['MCC']))
print('Precision/PPV: {:.3f}'.format(metrics['precision']))
print('NPV: {:.3f}'.format(metrics['NPV']))
