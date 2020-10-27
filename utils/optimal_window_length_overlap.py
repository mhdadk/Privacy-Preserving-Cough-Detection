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

uncut_cough_dir = '../../data_archive/data_audio/1_cough'
test_file = 'audioset_1.wav'

x,sr = librosa.load(path = os.path.join(uncut_cough_dir,test_file),
                    sr = None)

# length of overlapping windows in seconds

window_length = 1.5 # seconds

# length of overlap in seconds. This should be bigger than the max length
# of a cough

overlap_length = 1 # seconds
overlap = overlap_length / window_length # overlap fraction

# number of overlapping windows in the signal. This is equal to the length
# of the signal in seconds divided by the length of the overlapping window
# in seconds

num_overlap_windows = int(np.floor(x.shape[0]/(overlap_length*sr)))

rms_vals = np.zeros((int(num_overlap_windows/2),))
entropy_vals = np.zeros((int(num_overlap_windows/2),))
prev_rms = 0
prev_entropy = 0

# compute absolute difference in entropy and RMS between successive frames

for i in range(num_overlap_windows):
    
    start = int(i * overlap * window_length * sr)
    end = start + int(window_length * sr)
    
    frame = x[start:end]
    
    rms,entropy = get_stats(frame,sr)
    
    # compare every two successive frames
    
    if i % 2:
        
        rms_diff = abs(rms - prev_rms)
        entropy_diff = abs(entropy - prev_entropy)
        
        rms_vals[int(np.floor(i/2))] = rms_diff
        entropy_vals[int(np.floor(i/2))] = entropy_diff
        
    # save frame and stats
    
    prev_frame = frame.copy()
    prev_rms = rms
    prev_entropy = entropy
