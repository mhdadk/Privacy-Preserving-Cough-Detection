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
    
    for i in range(num_frames+1):
        frame = x[i*frame_width:(i+1)*frame_width]
        rms = np.sqrt(np.mean(frame**2))
        window = scipy.signal.windows.hann(frame.shape[0])
        y = np.multiply(frame,window)
        Y = np.abs(scipy.fft.fft(y))
        Y = Y/Y.sum() # make probability dist
        entropy = -np.sum(np.multiply(Y,np.log(Y)))
    

