import librosa
import numpy as np
import scipy
import os
import soundfile as sf

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

# where audio files are stored and where audio snippets should be written

src_dir = '../../../../datasets/1'
dst_dir = '../../../../datasets/8'
# src_dir = '../../data_archive/data_uncropped/speech'
# files = os.listdir(src_dir)
# dst_dir = '../../data/speech_cropped'
sample_rate = 16000

# initialize random number generator

rng = np.random.default_rng()

# how long snippets should be in seconds

snippet_length = 1.5

for dataset in os.listdir(src_dir):
    if dataset == '1_COUGH':
        continue
    
    for file in os.listdir(os.path.join(src_dir,dataset)):
        
        # get the total length of the audio file
        
        file_length_sec = librosa.get_duration(
                                filename = os.path.join(src_dir,
                                                        dataset,
                                                        file))
        
        # skip files that are shorter than snippet_length
        
        if file_length_sec < snippet_length:
            continue
        
        # check how many small snippets of audio have successfully been
        # admitted and written to disk
        
        num_passed = 0
        
        # to make sure that the same snippet of audio in an audio file is not
        # admitted twice, record the previous offset values. The offset
        # values determine where in the audio file the snippet is taken
        # from
        
        prev_offsets = []
        
        # sometimes, only 1 or 2 short sounds exist in a long audio file. In
        # this case, since there is a constraint that the same snippet is not
        # extracted twice, then it is not possible to extract more audio
        # snippets than there are. In that case, stop trying to search in
        # the audio file for snippets after loop_count has passed
        
        loop_count = 0
        
        # start randomly searching for snippets until 3 audio snippets have
        # successfully been admitted and written to disk. Otherwise, stop
        # searching if 100 iterations pass
        
        while (num_passed < 5 and loop_count < 100):
            
            # while loop counter
            
            loop_count += 1
            
            # uniformly sample an offset value between 0 and 1 second before
            # the end of the audio file. This is because a snippet is
            # 1 second long
            
            offset = rng.uniform(0.0,
                                 file_length_sec - snippet_length,
                                 1)[0]
            
            # used to indicate whether to continue to the next loop in this
            # while loop if the offset that is sampled above is within 1
            # second of previous offsets that were sampled and successfully
            # admitted
            
            next_loop = False
            
            # check if the current offset that was sampled above is within
            # 1 second of the previous offsets that were sampled and
            # previously admitted. Since an audio snippet is 1 second long,
            # then it makes sense to restrict offsets to be at least 1 second
            # away from each other so that overlapping snippets do not occur
            
            for j in prev_offsets:
                if abs(offset - j) < snippet_length:
                    next_loop = True
                    break
            
            if next_loop:
                continue
            
            # load snippet_length second audio snippet from file
            
            x,sr = librosa.load(path = os.path.join(src_dir,dataset,file),
                                sr = None,
                                mono = True,
                                offset = offset,
                                duration = snippet_length)
            
            # compute mean rms and mean entropy values
            
            rms_mean,entropy_mean = get_stats(x,sr)
            
            # these values were determined by estimation over a large number
            # of files. See check_thresholds.py for details. Note that the
            # 'or' can be changed to 'and' for better quality audio snippets.
            # However, this comes at the cost of increased computation and
            # less output
            
            if rms_mean > 0.035 or entropy_mean < 6.0:
                
                # indicate that 1 more audio snippet is successfully admitted
                
                num_passed += 1
                
                # record the offset that was used to produce this snippet
                
                prev_offsets.append(offset)
                
                # write the audio snippet to disk
                
                root,_ = os.path.splitext(file)
                
                sf.write(file = os.path.join(dst_dir,dataset,
                                             root+'_'+str(num_passed)+'.wav'),
                         data = x,
                         samplerate = sr)
