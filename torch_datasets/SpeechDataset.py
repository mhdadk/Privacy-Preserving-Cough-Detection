import pathlib
import torch
import torchaudio

class SpeechDataset(torch.utils.data.Dataset):
    
    def __init__(self,speech_data_dir,window_length,sample_rate,mode):
        
        # path to speech data folder
        
        self.raw_data_dir = pathlib.Path(speech_data_dir)
        
        # length of each speech window in seconds
        
        self.window_length = window_length
        
        # paths to speech files
        
        self.paths = list(pathlib.Path(speech_data_dir).iterdir())
        
        # new sample rate
        
        self.new_sr = sample_rate
    
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self,idx):
        
        # load window
        
        path = self.paths[idx]
        x,sr = torchaudio.load(filepath = path)
        
        # convert to mono
        
        x = torch.mean(x,dim=0,keepdim=True)
        
        # resample to self.sample_rate
        
        x = torchaudio.transforms.Resample(sr,self.new_sr)(x)
        
        return x
            
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import sounddevice as sd
    import random
    
    speech_data_dir = '../../datasets2/4/2_LIBRISPEECH'
    window_length = 1.5 # seconds
    dataset = SpeechDataset(speech_data_dir,window_length)
    
    idx = random.randint(0,len(dataset))
    x = dataset[idx]
    
    print('\nShowing signal for:\n{}'.format(dataset.paths[idx]))
    plt.plot(x.numpy()[0])
    sd.play(x.numpy()[0],16000)
