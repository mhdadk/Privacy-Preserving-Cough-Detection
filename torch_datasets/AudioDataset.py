import os
import torch
import torchaudio
import pandas as pd

class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self,dataset_dir,dataset_split_dir,mode,sample_rate):
        
        # path to dataset
        
        self.dataset_dir = dataset_dir
        
        # file paths are in the first column, while the corresponding label
        # is in the second column
        
        self.paths = pd.read_csv(os.path.join(dataset_split_dir,
                                              'data_'+mode+'.csv'),
                                 header = None)
        
        # what sampling rate to resample audio to
        
        self.sample_rate = sample_rate
    
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self,idx):
        
        # load audio file and its sample rate
        
        path = os.path.join(self.dataset_dir,self.paths.iloc[idx,0])
        x,sr = torchaudio.load(filepath = path)
        
        # resample to self.sample_rate
        
        x = torchaudio.transforms.Resample(sr,self.sample_rate)(x)
        
        # convert to mono
        
        x = torch.mean(x,dim=0,keepdim=True)
        
        # get audio signal label
        
        label = self.paths.iloc[idx,1]
        
        return x,label
            
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import sounddevice as sd
    import random
    
    dataset_dir = '../../datasets/3'
    dataset_split_dir = '../../datasets_splits/3'
    mode = 'train'
    sample_rate = 16000
    dataset = AudioDataset(dataset_dir,dataset_split_dir,mode,sample_rate)
    
    idx = random.randint(0,len(dataset))
    x,label = dataset[idx]
    
    # compute log Mel spectrogram
    
    log = torchaudio.transforms.AmplitudeToDB()
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = sample_rate,
                                                    n_fft = 1024,
                                                    n_mels = 64,
                                                    hop_length = 10)
    log_mel_spec = log(mel_spec(x))
    
    # print('\nShowing signal for:\n{}'.format(dataset.paths[idx]))
    # plt.plot(x.numpy())
    # sd.play(x.numpy(),sample_rate)
