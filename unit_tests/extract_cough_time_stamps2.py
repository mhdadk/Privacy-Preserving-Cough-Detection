import os
import numpy as np
import librosa

uncut_cough_dir = '../../../data_archive/data_audio/1_cough'
cut_cough_dir = '../../../data/cough'
test_cough = 'audioset_2_3.wav'
substrings = test_cough.split('_')[:2]
test_file = substrings[0]+'_'+substrings[1]+'.wav'

x,sr = librosa.load(path = os.path.join(uncut_cough_dir,test_file),
                    sr = None)

template = librosa.load(path = os.path.join(cut_cough_dir,test_cough),
                        sr = None)[0]

def get_start_and_end(x,template):
    
    # used to ignore correlations involving large values in x
    
    threshold = np.correlate(template,template,'valid')[0]
    
    # cross-correlate template with signal
    
    signal = np.correlate(x,template,'same')
    
    # set values that are too large to 0
    
    signal[signal>threshold] = 0
    
    # find the center of the template in the signal
    
    center = np.argmax(signal)
    
    # if length of template is odd
    
    template_length = template.shape[0]
    
    if template_length % 2:
        
        start = center - int(np.floor(template_length/2))
        end = center + int(np.floor(template_length/2))
        
    else: # if length of template is even
        
        start = center - int(template_length/2)
        end = center + int(template_length/2) - 1
    
    return start,end

start,end = get_start_and_end(x,template)

# import sounddevice as sd
# sd.play(template,sr)
# sd.play(x[start:end+1],sr)
