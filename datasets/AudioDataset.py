import os
import pickle
import torch
import torchaudio

class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self,data_dir,data_split_dir,sample_rate,net_type,mode):
        
        # where the data is stored
        
        self.data_dir = data_dir
        
        """
        Where the pickled lists containing paths to the training, validation,
        and testing files are stored
        """
        
        self.data_split_dir = data_split_dir
        
        # what sampling rate to resample audio to
        
        self.sample_rate = sample_rate
        
        """
        If this data is for the discrimination 'disc' network,
        reconstruction 'recon' network, or 'both'
        """
        
        self.net_type = net_type
        
        """
        training 'train', validation 'val', or testing 'test' mode
        """
        
        self.mode = mode
        
        # what the label numbers mean
        
        self.label_map = {0:'other',
                          1:'cough',
                          2:'speech'}
        
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
        
        if self.net_type == 'recon':
            self.paths = self.splits['speech_'+self.mode]
        else:
            self.paths = (self.splits['other_'+self.mode] + 
                          self.splits['cough_'+self.mode] + 
                          self.splits['speech_'+self.mode])
            self.labels = ([0]*len(self.splits['other_'+self.mode]) +
                           [1]*len(self.splits['cough_'+self.mode]) +
                           [2]*len(self.splits['speech_'+self.mode]))
    
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self,idx):
        
        # if reconstruction network is of interest, speech audio is only
        # available
        
        if self.net_type == 'recon':
            label = 2
        else:
            label = self.labels[idx]
        
        # load audio file and its sample rate
        
        x,sr = torchaudio.load(
                    filepath = os.path.join(self.data_dir,
                                            self.label_map[label],
                                            self.paths[idx]))
        
        # convert to mono
        
        x = torch.mean(x,dim=0)
        
        # crop to 1 second if necessary
        
        x = x[:sr]            
        
        # resample to 16 kHz
        
        x = torchaudio.transforms.Resample(sr,self.sample_rate)(x)
        
        # pad cough signals with zeros to make them 1 second long
        
        if label == 1:
            pad_length = self.sample_rate - x.shape[0]
            x = torch.nn.functional.pad(input = x,
                                        pad = (0,pad_length),
                                        mode = 'constant',
                                        value = 0)
        
        # put tensors on GPU if available
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        
        return (x,label)
            
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import sounddevice as sd
    import random
    
    data_dir = '../../data'
    data_split_dir = '../../data_split'
    sample_rate = 16000
    net_type = 'recon'
    mode = 'val'
    dataset = AudioDataset(data_dir,
                           data_split_dir,
                           sample_rate,
                           net_type,
                           mode)
    
    idx = random.randint(0,len(dataset))
    x,label = dataset[idx]
    
    print('\nShowing signal for:\n{}'.format(os.path.join(
                                                data_dir,
                                                dataset.label_map[label],
                                                dataset.paths[idx])))
    plt.plot(x.numpy())
    sd.play(x.numpy(),sample_rate)
