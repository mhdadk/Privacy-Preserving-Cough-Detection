import numpy as np
import librosa
import scipy
import os

data_dir = '../../data/cough'

files = os.listdir(data_dir)

# rms in first column, entropy in second column
stats = np.zeros((len(files),2))

for i,file in enumerate(files):
    x,sr = librosa.load(path = os.path.join(data_dir,file),
                        sr = None,
                        mono = True)
    
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
