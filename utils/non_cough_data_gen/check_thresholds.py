import librosa
import numpy as np
import scipy
import os

def get_stats(x,sr,frame_sample_length=0.064):
    
    # frame width in seconds
    
    frame_width = int(frame_sample_length * sr)
    
    # compute number of frames in cough signal
    
    num_frames = int(np.floor(x.shape[0]/frame_width))
    
    # to store rms and entropy values for each frame
    
    frames_rms = []
    frames_entropy = []
    
    """
    for each frame in the signal, compute the rms value
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
        
        # compute magnitude of discrete Fourier transform. The 1e-10 is
        # added to avoid dividing by 0
        
        Y = np.abs(scipy.fft.fft(y)) + 1e-10
        
        # normalize to make it a probability distribution
        
        Y = Y/Y.sum()
        
        # compute entropy and save the value
        
        entropy = -np.sum(np.multiply(Y,np.log(Y)))
        frames_entropy.append(entropy)
    
    # compute the average rms for all frames (except the last one)
    # in the cough signal
    
    frames_rms_mean = sum(frames_rms)/len(frames_rms)
    frames_entropy_mean = sum(frames_entropy)/len(frames_entropy)
    
    return frames_rms_mean,frames_entropy_mean

# check thresholds for cough signals

cough_dir = '../../data/cough'
cough_files = os.listdir(cough_dir)

# to store the average rms and entropy values for each cough
# signal. Average rms values are saved in the first column while
# average entropy values are saved in the second column

stats = np.zeros((len(cough_files),2))

for i,file in enumerate(cough_files):
    x,sr = librosa.load(path = os.path.join(cough_dir,file),
                        sr = None,
                        mono = True)
    stats[i,:] = get_stats(x,sr)
    
cough_rms_mean = stats[:,0].mean()
cough_rms_var = stats[:,0].var(ddof=1)
cough_entropy_mean = stats[:,1].mean()
cough_entropy_var = stats[:,1].var(ddof=1)

# check thresholds for first 2000 speech signals

speech_dir = '../../data/speech'
speech_files = os.listdir(speech_dir)[:2000]

# to store the average rms and entropy values for each cough
# signal. Average rms values are saved in the first column while
# average entropy values are saved in the second column

stats = np.zeros((len(speech_files),2))

for i,file in enumerate(speech_files):
    x,sr = librosa.load(path = os.path.join(speech_dir,file),
                        sr = None,
                        mono = True)
    stats[i,:] = get_stats(x,sr)

speech_rms_mean = stats[:,0].mean()
speech_rms_var = stats[:,0].var(ddof=1)
speech_entropy_mean = stats[:,1].mean()
speech_entropy_var = stats[:,1].var(ddof=1)

# check thresholds for 2000 noisy signals

# to store the average rms and entropy values for each cough
# signal. Average rms values are saved in the first column while
# average entropy values are saved in the second column

stats = np.zeros((2000,2))

for i in range(2000):
    x = np.random.normal(0,1,100000)
    sr = 44100
    stats[i,:] = get_stats(x,sr)

noise_rms_mean = stats[:,0].mean()
noise_rms_var = stats[:,0].var(ddof=1)
noise_entropy_mean = stats[:,1].mean()
noise_entropy_var = stats[:,1].var(ddof=1)
