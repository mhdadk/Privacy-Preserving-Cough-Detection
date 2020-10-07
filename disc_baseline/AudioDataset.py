import os
import pickle
import torch
import torchaudio

class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self,net_type,data_split_dir,sample_rate,mode):
        
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
                # file[:-4] is used to discard the 'pkl' substring at the end 
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
            self.paths = (self.splits['other_'+self.mode] + 
                          self.splits['cough_'+self.mode])
            self.labels = ([0]*len(self.splits['other_'+self.mode]) + 
                           [1]*len(self.splits['cough_'+self.mode]))
        else: # reconstruction network
            self.paths = self.splits['speech_'+self.mode]
            self.labels = [0]*len(self.splits['speech_'+self.mode])
    
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self,idx):
        
        # get signal label
        
        label = self.labels[idx]
        
        # load audio file and its sample rate
        
        x,sr = torchaudio.load(filepath = self.paths[idx])
        
        # convert to mono
        
        x = torch.mean(x,dim=0)
        
        # crop to 1 second if necessary
        
        x = x[:sr]            
        
        # resample to self.sample_rate
        
        x = torchaudio.transforms.Resample(sr,self.sample_rate)(x)
        
        # pad signals with zeros to make sure they are 1 second long
        
        pad_length = self.sample_rate - x.shape[0]
        x = torch.nn.functional.pad(input = x,
                                    pad = (0,pad_length),
                                    mode = 'constant',
                                    value = 0)
        
        return (x,label)
            
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import sounddevice as sd
    import random
    
    net_type = 'recon'
    data_split_dir = '../../data_split'
    sample_rate = 16000
    mode = 'val'
    dataset = AudioDataset(net_type,data_split_dir,sample_rate,mode)
    
    idx = random.randint(0,len(dataset))
    x,label = dataset[idx]
    
    print('\nShowing signal for:\n{}'.format(dataset.paths[idx]))
    plt.plot(x.numpy())
    sd.play(x.numpy(),sample_rate)
