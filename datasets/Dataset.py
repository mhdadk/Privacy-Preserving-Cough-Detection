import torch

from torch import nn

import os

import random

import librosa

import torchaudio

import torchvision as tv

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
        compute the average root mean square (rms) and entropy values for
        each frame of cough signals. These provide thresholds to be used
        to control the admission of audio that is cropped from speech and
        other files. More precisely, if a majority of frames satisfy these
        thresholds, then the audio snippet is admitted. Otherwise, it is
        discarded.
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
    
    def _get_cough_length_stats(self):
        
        # list of cough files
        
        files = self.splits['cough_'+self.mode]
        
        # to store the length of each cough
        
        lengths = np.zeros((len(files),))
        
        # record the length of each cough
        
        for i,file in enumerate(files):
            lengths[i] = librosa.get_duration(filename = file)
        
        # compute the mean and variance of the lengths of each cough.
        # These are unbiased maximum-likelihood estimates
        
        self.cough_length_mean = lengths.mean()
        self.cough_length_var = lengths.var(ddof=1)
    
    def _get_cough_frame_stats(self):
        
        """
        for each cough signal, split it into 64 ms frames. Then, for each
        frame, compute the root mean square and entropy values. Entropy
        values are computed by normalizing the discrete Fourier transform
        of the samples in a frame and the result is treated as a
        probability distribution.
        
        See section 4.1 in the following paper for details:
            
        "SoundSense: Scalable Sound Sensing for People-Centric
        Applications on Mobile Phones"
        
        Available here:
            
        https://pac.cs.cornell.edu/pubs/mobisys09_soundsense.pdf
        
        """
        
        # list of cough files
        
        files = self.splits['cough_'+self.mode]
        
        # to store the average rms and entropy values for each cough
        # signal. Average rms values are saved in the first column while
        # average entropy values are saved in the second column
        
        stats = np.zeros((len(files),2))     
        
        for i,file in enumerate(files):
            
            # load cough signal
            
            x,sr = librosa.load(path = file,sr = None,mono = True)
            
            # frames are 64 ms wide
            
            frame_width = int(0.064 * sr)
            
            # compute number of frames in cough signal
            
            num_frames = int(np.floor(x.shape[0]/frame_width))
            
            # to store rms and entropy values for each frame in a cough
            # signal
            
            frames_rms = []
            frames_entropy = []
            
            """
            for each frame in the cough signal, compute the rms value
            of that frame and the entropy of that frame. These values
            are then appended to the rms_frames and entropy_frames above.
            
            Note that since the average of these frame values will
            eventually be computed, then frames with a length shorter than
            frame_width would be given the same weight as frames with a
            length of frame_width, which would lead to biased estimates
            of the average of the rms and entropy values of these frames.
            Therefore, frames shorter than frame_width are ignored.
            
            For example, suppose that a frame has a length of 4500 samples,
            and frame_width is 1024 samples. This means that there are
            4500/1024 = 4.395... frames. So, the first 4 frames of lengths
            1024 samples will be considered. The final frame with a length
            of 4500 - (1024*4) = 404 samples will not be considered.
            """
            
            for j in range(num_frames):
                
                # extract a frame
                
                frame = x[j*frame_width:(j+1)*frame_width]
                
                # compute rms and save the value
                
                rms = np.sqrt(np.mean(frame**2))
                frames_rms.append(rms)
                
                # apply a hanning window to mitigate spectral leakage
                
                window = scipy.signal.windows.hann(frame.shape[0])
                y = np.multiply(frame,window)
                
                # compute magnitude of discrete Fourier transform
                
                Y = np.abs(scipy.fft.fft(y))
                
                # normalize to make it a probability distribution
                
                Y = Y/Y.sum()
                
                # compute entropy and save the value
                
                entropy = -np.sum(np.multiply(Y,np.log(Y)))
                frames_entropy.append(entropy)
            
            # compute the average rms for all frames (except the last one)
            # in the cough signal
            
            frames_rms_mean = sum(frames_rms)/len(frames_rms)
            frames_entropy_mean = sum(frames_entropy)/len(frames_entropy)
            
            # save these average values
            
            stats[i,0] = frames_rms_mean
            stats[i,1] = frames_entropy_mean
            
        self.rms_threshold = stats[:,0].mean()
        self.entropy_threshold = stats[:,1].mean()
    
    def __len__(self):
        return len(self.paths)
    
    def _sample_cough_length(self):
        
        x = self.rng.normal(self.cough_length_mean,
                            np.sqrt(self.cough_length_var),
                            1)
        
        # cough length can't be negative or greater than 1. If it is,
        # sample from uniform distribution.
        
        if x < 0 or x >= 1:
            
            x = self.rng.uniform(0.0,1.0,1)
            
        return x
    
    def _check_audio_snippet(self,audio,frame_length=0.064,
                             majority_ratio=0.8):
        
        """
        Split short (< 1 second) audio snippets into frames of length frame_length.
        Then, for each frame, compute the root mean square (rms) and
        entropy values. Entropy values are computed by normalizing the
        discrete Fourier transform of the samples in a frame and the result
        is treated as a probability distribution. Next, if majority_ratio
        of frames achieve rms values above the threshold of:
            
        self.rms_threshold
        
        And achieve entropy values below the threshold of:
            
        self.entropy_threshold
        
        See section 4.1 in the following paper for details:
            
        "SoundSense: Scalable Sound Sensing for People-Centric
        Applications on Mobile Phones"
        
        Available here:
            
        https://pac.cs.cornell.edu/pubs/mobisys09_soundsense.pdf
        
        """
        
        # list of cough files
        
        files = self.splits['cough_'+self.mode]
        
        # to store the average rms and entropy values for each cough
        # signal. Average rms values are saved in the first column while
        # average entropy values are saved in the second column
        
        stats = np.zeros((len(files),2))     
        
        for i,file in enumerate(files):
            
            # load cough signal
            
            x,sr = librosa.load(path = file,sr = None,mono = True)
            
            # frames are 64 ms wide
            
            frame_width = int(0.064 * sr)
            
            # compute number of frames in cough signal
            
            num_frames = int(np.floor(x.shape[0]/frame_width))
            
            # to store rms and entropy values for each frame in a cough
            # signal
            
            frames_rms = []
            frames_entropy = []
            
            """
            for each frame in the cough signal, compute the rms value
            of that frame and the entropy of that frame. These values
            are then appended to the rms_frames and entropy_frames above.
            
            Note that since the average of these frame values will
            eventually be computed, then frames with a length shorter than
            frame_width would be given the same weight as frames with a
            length of frame_width, which would lead to biased estimates
            of the average of the rms and entropy values of these frames.
            Therefore, frames shorter than frame_width are ignored.
            
            For example, suppose that a frame has a length of 4500 samples,
            and frame_width is 1024 samples. This means that there are
            4500/1024 = 4.395... frames. So, the first 4 frames of lengths
            1024 samples will be considered. The final frame with a length
            of 4500 - (1024*4) = 404 samples will not be considered.
            """
            
            for j in range(num_frames):
                
                # extract a frame
                
                frame = x[j*frame_width:(j+1)*frame_width]
                
                # compute rms and save the value
                
                rms = np.sqrt(np.mean(frame**2))
                frames_rms.append(rms)
                
                # apply a hanning window to mitigate spectral leakage
                
                window = scipy.signal.windows.hann(frame.shape[0])
                y = np.multiply(frame,window)
                
                # compute magnitude of discrete Fourier transform
                
                Y = np.abs(scipy.fft.fft(y))
                
                # normalize to make it a probability distribution
                
                Y = Y/Y.sum()
                
                # compute entropy and save the value
                
                entropy = -np.sum(np.multiply(Y,np.log(Y)))
                frames_entropy.append(entropy)
            
            # compute the average rms for all frames (except the last one)
            # in the cough signal
            
            frames_rms_mean = sum(frames_rms)/len(frames_rms)
            frames_entropy_mean = sum(frames_entropy)/len(frames_entropy)
            
            # save these average values
            
            stats[i,0] = frames_rms_mean
            stats[i,1] = frames_entropy_mean
            
        self.rms_threshold = stats[:,0].mean()
        self.entropy_threshold = stats[:,1].mean()
        
    def __getitem__(self,idx):
        
        # if reconstruction network is of interest, speech audio only
        # is available
        
        if self.net_type == 'recon':
            label = 2
        else:
            label = self.labels[idx]
        
        """
        if not cough signal, then sample snippet of audio with length that
        is sampled from normal distribution and location that is sampled
        from uniform distribution.
        """
        
        if label != 1:
        
            sample_length_sec = self._sample_cough_length()
            file_length_sec = librosa.get_duration(
                                    filename = self.paths[idx])
            offset = self.rng.uniform(0.0,
                                      file_length_sec - sample_length_sec,
                                      1)
            x,_ = librosa.load(path = self.paths[idx],
                               sr = self.sample_rate,
                               mono = True,
                               offset = offset,
                               duration = sample_length_sec)
            
        else: # if cough
            
            x,_ = librosa.load(path = self.paths[idx],
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
