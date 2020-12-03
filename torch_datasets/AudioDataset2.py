import os
import pickle
import torch
import torchaudio

from FENet import FENet

class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self,net_type,data_split_dir,sample_rate,mode,device):
        
        self.device = device
        param_path = 'mx-h64-1024_0d3-1.17.pkl'
        self.feature_extractor = FENet(param_path).to(device)
        # freeze weights of feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        """
        whether to sample cough and other data, or only speech data
        """
        
        self.net_type = net_type
        
        """
        Where the pickled lists containing paths to the training, validation,
        and testing files are stored
        """
        
        self.data_split_dir = data_split_dir
        
        # what sampling rate to resample audio to
        
        self.sample_rate = sample_rate
        
        """
        training 'train', validation 'val', or testing 'test' mode
        """
        
        self.mode = mode
        
        """
        get paths to the training, validation, and testing files
        """
        
        self._get_file_paths()
        
    def _get_file_paths(self):
        
        # to store each un-pickled list
        
        self.splits = {}
        
        # for each pickled file
        
        for file in os.listdir(self.data_split_dir):
            with open(os.path.join(self.data_split_dir,file),'rb') as fp:
                # file[:-4] is used to discard the '.pkl' substring at the end 
                self.splits[file[:-4]] = pickle.load(fp)
                
        """
        depending on which network will be trained, validated, or tested,
        assign the corresponding paths and labels. If the reconstruction
        network is of interest, then only the speech signals are provided.
        If the discrimination network is of interest, then all signals are
        provided. If both networks are of interest, then all signals are
        provided. Note that the reconstruction network does not need labels
        """
        
        if self.net_type == 'disc': # discriminator network
            self.paths = (self.splits['other_long_'+self.mode] + 
                          self.splits['cough_long_'+self.mode])
            self.labels = ([0]*len(self.splits['other_long_'+self.mode]) + 
                           [1]*len(self.splits['cough_long_'+self.mode]))
        else: # reconstruction network
            self.paths = self.splits['speech_long_'+self.mode]
            self.labels = [0]*len(self.splits['speech_long_'+self.mode])
    
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self,idx):
        
        # get signal label
        
        label = self.labels[idx]
        
        # load audio file and its sample rate
        
        x,sr = torchaudio.load(filepath = self.paths[idx])
        
        # resample to self.sample_rate
        
        x = torchaudio.transforms.Resample(sr,self.sample_rate)(x)
        
        # convert to mono
        
        x = torch.mean(x,dim=0,keepdim=True).to(self.device)
        
        # log mel spectrogram
        
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = 16000,
                                                    n_fft = 1024,
                                                    n_mels = 128,
                                                    hop_length = 64).to(self.device)
        to_dB = torchaudio.transforms.AmplitudeToDB().to(self.device)
        images = to_dB(mel_spec(x))
        
        # compute 1024-dimensional feature vectors
        
        x = self.feature_extractor(images.unsqueeze(dim=0))
        
        return (x,label)
            
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import sounddevice as sd
    import random
    
    net_type = 'disc'
    data_split_dir = 'data_split'
    sample_rate = 16000
    mode = 'val'
    dataset = AudioDataset(net_type,data_split_dir,sample_rate,mode,'cuda')
    
    idx = random.randint(0,len(dataset))
    x,label = dataset[idx]
    
    print('\nShowing signal for:\n{}'.format(dataset.paths[idx]))
    # plt.plot(x.numpy())
    # sd.play(x.numpy(),sample_rate)
