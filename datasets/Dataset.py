import torch

from torch import nn

import os

import random

import librosa

import torchaudio

import torchvision as tv

from sklearn.model_selection import train_test_split

import scipy

import numpy as np

import pickle

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self,data_split_dir,sample_rate,net_type,mode):
        
        """
        training 'train', validation 'val', or testing 'test' mode
        """
        
        self.mode = mode
        
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
        get paths to the training, validation, and testing files
        """
        
        self._get_file_paths()
        
        # initialize random number generator
        
        self.rng = np.random.default_rng()
        
        """
        compute the mean and variance of the lengths of coughs. These will be
        used to sample lengths of audio to crop out of speech and other audio
        files
        """
        
        self._get_cough_length_stats()
        
        """
        compute the mean root mean square and entropy values for each frame of
        cough signals. These provide thresholds to be used to control the
        admission of audio that is cropped from speech and other files. More
        precisely, if a majority of frames satisfy these thresholds, then
        the audio snippet is admitted. Otherwise, it is discarded.
        """
        
        self._get_cough_frame_stats()
        
        
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
        network is of interest, then 
        """
        
        if net_type == 'recon':
            self.paths = self.splits['speech_'+self.mode]
        else:
            self.paths = (self.splits['other_'+self.mode] + 
                          self.splits['cough_'+self.mode] + 
                          self.splits['speech_'+self.mode])
            self.labels = ([0]*len(self.splits['other_'+self.mode] +
                           [1]*len(self.splits['cough_'+self.mode] +
                           [2]*len(self.splits['speech_'+self.mode])
        
    def _get_cough_length_stats(self):

        files = self.splits['cough_'+self.mode]
        
        lengths = np.zeros((len(files),))
        
        for i,file in enumerate(files):
            lengths[i] = librosa.get_duration(filename = file)
        
        self.cough_length_mean = lengths.mean()
        self.cough_length_var = lengths.var(ddof=1)
    
    def _get_cough_frame_stats(self):
        
        files = self.splits['cough_'+self.mode]
        
        # rms in first column, entropy in second column
        # stats = np.zeros((len(files),2))
        
        for i,file in enumerate(files):
            
            x,sr = librosa.load(path = file,sr = None,mono = True)
            
            # frames are 64 ms wide
            
            frame_width = int(0.064 * sr)
            
            num_frames = int(np.floor(x.shape[0]/frame_width))
            
            # note that for a numpy array b, b[x:infinity] is the same
            # as b[x:]
            
            rms_val = []
            entropies = []
            
            # frames with length shorter than frame_width are given same weight as
            # frames with length of frame_width, which will lead to a biased rms value
            
            for j in range(num_frames):
                frame = x[j*frame_width:(j+1)*frame_width]
                rms = np.sqrt(np.mean(frame**2))
                rms_val.append(rms)
                window = scipy.signal.windows.hann(frame.shape[0])
                y = np.multiply(frame,window)
                Y = np.abs(scipy.fft.fft(y))
                Y = Y/Y.sum() # make probability dist
                entropy = -np.sum(np.multiply(Y,np.log(Y)))
                entropies.append(entropy)
                
            rms_mean = sum(rms_val)/len(rms_val)
            entropy_mean = sum(entropies)/len(entropies)
            
            stats[i,0] = rms_mean
            stats[i,1] = entropy_mean
            
        print(stats.mean(axis=0))
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,idx):
        
        # assign labels for cough and non-cough files
        
        label = self.labels[idx]
        
        # load waveform
                
        waveform,_ = librosa.load(path = self.paths[idx],
                                  sr = self.sample_rate,
                                  mono = True)
        
        # if the audio file is shorter than 9 seconds, pad it with zeros to be
        # 10 seconds long, and if it is longer than 11 seconds, decimate it to
        # be 10 seconds long
        
        if (waveform.shape[0] < 9*self.sample_rate):
            
            pad_size = (10*self.sample_rate) - waveform.shape[0]
            
            waveform = np.pad(waveform,(0,pad_size))
        
        elif (waveform.shape[0] > 11*self.sample_rate):
            
            waveform = scipy.signal.resample(waveform,
                                             10*self.sample_rate,
                                             window = 'hann')
        
        # to put tensors on GPU if available
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # convert to tensor from numpy array
        
        waveform = torch.tensor(waveform).to(device)
        
        # compute spectrogram using short-time Fourier transform
        
        spec_func = torchaudio.transforms.Spectrogram(
                    n_fft = 1024,
                    hop_length = 56,
                    power = 2.0,
                    normalized = True)
        
        # convert frequencies to Mel scale
        
        mel_func = torchaudio.transforms.MelScale(
                   n_mels = 224,
                   sample_rate = self.sample_rate)
        
        # dilate small values
        
        log_func = torchaudio.transforms.AmplitudeToDB(stype = 'power')
        
        log_mel_spec = log_func(mel_func(spec_func(waveform)))

        # convert to RGB image
        
        log_mel_spec = torch.unsqueeze(log_mel_spec,dim=0)
        
        log_mel_spec = torch.cat(3*[log_mel_spec],dim=0)
        
        # downsample image to fit input to network. nn.functional.interpolate()
        # expects B x C x H x W dimensions for input image.
        
        log_mel_spec = torch.unsqueeze(log_mel_spec,dim=0)
        
        log_mel_spec = nn.functional.interpolate(
                       log_mel_spec,
                       size = (224,224),
                       mode = 'bilinear',
                       align_corners = False)
        
        log_mel_spec = torch.squeeze(log_mel_spec,dim=0)
        
        # standardization
        
        normalize_func = tv.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
        
        log_mel_spec = normalize_func(log_mel_spec)
        
        return (log_mel_spec,label)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    
    from playsound import playsound
    
    data_dir = '../../data'
    
    sample_rate = 22050 # Hz
    
    mode = 'test'
    
    dataset = AudioDataset(data_dir,sample_rate,mode)
    
    idx = random.randint(0,len(dataset))
    
    image,label = dataset[idx]
    
    print('\nShowing spectrogram for:\n{}'.format(dataset.paths[idx]))
    
    plt.imshow(image[0])
        
    playsound(dataset.paths[idx])
