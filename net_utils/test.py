import torch
import torchaudio
from sklearn.metrics import confusion_matrix

def test(net,dataloader,device):
    
    # put in testing mode
    
    net.eval()
    
    # store class predictions
    
    class_preds = []
    
    labels_all = []
    
    for signals,labels in dataloader:
        
        # load onto GPU
        
        signals = signals.to(device).unsqueeze(dim=1)
        
        # store labels
        
        labels_all.extend(labels.tolist())
        
        # compute log Mel spectrogram
        
        images = torchaudio.transforms.MelSpectrogram(sample_rate = 16000,
                                             n_fft = 1024,
                                             n_mels = 256,
                                             hop_length = 63)(signals)
        images = torchaudio.transforms.AmplitudeToDB()(images)
        
        # don't compute grad_fn to conserve RAM
        
        with torch.set_grad_enabled(False):
        
            # outputs of net for batch input
            
            outputs = net(images).squeeze()
        
        # since sigmoid(0) = 0.5, then negative values correspond to class 0
        # and positive values correspond to class 1
        
        class_preds.extend((outputs > 0).squeeze().tolist())
    
    CM = confusion_matrix(labels_all,class_preds,labels=[0,1])
    
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
    
    metrics = {
        
        'CM':CM,
        'sens':sensitivity,
        'spec':specificity,
        'acc':accuracy,
        'bal_acc':balanced_accuracy,
        'MCC':MCC,
        'PPV':PPV,
        'NPV':NPV
        
    }
    
    return metrics
