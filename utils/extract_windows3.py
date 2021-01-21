import pathlib
import shutil
import csv

import numpy as np
import scipy as sp
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split

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
        
        window = sp.signal.windows.hann(frame.shape[0])
        y = np.multiply(frame,window)
        
        # compute magnitude of discrete Fourier transform. The 1e-10 is
        # added to avoid dividing by 0
        
        Y = np.abs(sp.fft.fft(y)) + 1e-10
        
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

# turn off SettingWithCopyWarning. default is 'warn'. Just in case,
# this should be set to 'warn' during testing

pd.options.mode.chained_assignment = None

# initialize random number generator

rng_seed = 42
rng = np.random.default_rng(seed = rng_seed)

# where the raw data is located

data_dir = pathlib.Path('../../data/raw')

# load paths

print('\nLoading paths...')

paths = []

for label in data_dir.iterdir():
    # need this to only split 0_LIRBISPEECH and 1_COUGH
    if (label.name != '0_LIBRISPEECH' and
        label.name != '1_COUGH'):
        continue
    for file in label.iterdir():
        paths.append('{}/{}'.format(file.parent.name,file.name))

paths = pd.Series(paths)

# load cough timestamps

print('\nLoading cough timestamps...')

cough_timestamps = {}
fp = open('cough_timestamps.csv')
csv_reader = csv.reader(fp,delimiter=',')
prev_filename = ''

for row in csv_reader:
    file1,file2,_ = row[0].split('_')
    filename = file1 + '_' + file2 + '.wav'
    cough_start = float(row[1])
    cough_end = float(row[2])
    if prev_filename == filename:
        cough_timestamps[filename].append([cough_start,cough_end])
    else:
        cough_timestamps[filename] = [[cough_start,cough_end]]
    prev_filename = filename

fp.close()

# split paths

print('\nSplitting paths...')

paths_train,paths_val = train_test_split(paths,
                                         train_size = 0.8,
                                         random_state = rng_seed,
                                         shuffle = True,
                                         stratify = paths.str[0].astype(int))

paths_val,paths_test = train_test_split(paths_val,
                                        train_size = 0.5,
                                        random_state = rng_seed,
                                        shuffle = True,
                                        stratify = paths_val.str[0].astype(int))

# how long extracted windows should be in seconds

window_lengths = [1.0, 1.5, 2.0, 2.5, 3.0]

# number of windows to extract per cough

windows_per_cough = 102

"""
for speech windows,start randomly searching for snippets until
<max_passed> windows have successfully been admitted
"""

max_passed = 4

# whether to show train, val, and test ratios and class split ratios

verbose = True

for window_length in window_lengths:
    
    print('\nProcessing windows of length {} seconds...'.format(window_length))
    
    # where the processed data will be stored in csv format

    dst_dir = data_dir.parent / (str(window_length).replace('.','-') + 's')
    
    """
    if the folders 1-0s, 1-5s,..., 3-0s already exist, then delete them
    and re-create them to start from scratch. Otherwise, create these
    folders
    """
    
    if dst_dir.exists():
        shutil.rmtree(dst_dir) # delete non-empty directories
    
    dst_dir.mkdir() # create the folder
    
    """
    to store the processing instructions for the training, validation,
    and testing data. The first list stores the training metadata, the
    second list stores the validation metadata, and the third list
    stores the testing metadata
    """
    
    data = [[],[],[]]
    
    # to count the number of processed files
    
    num_processed = 0
    
    for i,pathset in enumerate([paths_train,paths_val,paths_test]):
        for _,path in pathset.iteritems():
            
            # track progress
            
            print('\rProcessed {} files'.format(num_processed),
                  end='',flush=True)
            
            # full path to file
            
            full_path = data_dir / path
            
            # get file length
            
            file_length = librosa.get_duration(filename = str(full_path))
            
            """
            if the entire audio file is not at least as long as 1.5
            times the length of the window to be extracted, skip the
            audio file
            """
            
            if file_length < 1.5 * window_length:
                continue
            
            # get name of dataset folder and file name
            
            dataset_name,filename = path.split('/')
            
            """
            if analyzing the 1_COUGH dataset, then extract <num_windows>
            windows of length <window_length> seconds that contain the
            coughs 
            """
            
            if 'COUGH' in dataset_name:
                
                # get locations of coughs in file. Need exception handling
                # in case file does not contain coughs
                
                try:
                    coughs = cough_timestamps[filename]
                except KeyError:
                    continue
                
                """
                given the signal:
                    
                  !       $         *       ^
                0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0    
                
                Where 0's represent non-cough and 1's represent a cough, the 
                extracted window must be 10 samples long, and the $ symbol
                represents the first sample of the cough, then a 10-sample
                window can start from the 2nd sample (!) of the signal and end
                at the last sample of the cough (*), or it can start from the
                1st sample of the cough ($) and end at the 15th sample of the
                signal (^).
                
                The 10-sample window can range anywhere from these two
                extremes, with the condition that the window contain 100% of
                the cough.
                
                The start variable below is a uniformly random sample of the
                starting time of the window that can range from the ! sample,
                which is equal to the * sample minus the window length, to the
                starting sample of the cough, denoted by $. Note, however, that
                since the difference between * and the window length can be
                negative, then the max operator is needed to keep this
                difference greater than or equal to 0
                
                Also, to avoid windows that are less than <window_length>
                seconds, the min operator is needed for the end time of
                the window
                """
                
                for cough_start,cough_end in coughs:
                    
                    """
                    if the length of the cough is greater than the length
                    of the window to be extracted, skip this cough
                    """
                    
                    if (cough_end - cough_start) > window_length:
                        continue
                    
                    # range for possible starting index of window
                    
                    low = max(0,cough_end - window_length)
                    high = min(cough_start,file_length - window_length)
                    
                    # extract <windows_per_cough> windows around this cough
                    
                    for _ in range(windows_per_cough):
                        
                        # track progress
                        
                        num_processed += 1
                        
                        # sample a random starting time for the window
                        
                        start = rng.uniform(low = low, high = high)
                        
                        # store the metadata
                        
                        data[i].append([path,start,start + window_length])
            
            else:
                    
                # number of windows to be admitted
            
                num_passed = 0
                
                """
                sometimes, only 1 or 2 short sounds exist in a long audio file.
                In this case, stop searching after <iter_count> iterations has
                passed
                """
                
                iter_count = 0
                
                while (num_passed < max_passed and iter_count < 100):
                    
                    iter_count += 1
                    
                    """
                    uniformly sample a starting time for the window
                    between 0 and <file_length - window_length> seconds. This
                    is to ensure that the window does not exceed the end of
                    the audio file
                    """
                    
                    start = rng.uniform(low = 0.0,
                                        high = file_length - window_length)
                    
                    # load the window from the audio file to check if it
                    # can be admitted
                    
                    x,sr = librosa.load(path = str(full_path),
                                        sr = None,
                                        mono = True,
                                        offset = start,
                                        duration = window_length)
                    
                    # compute mean rms and mean entropy values
                    
                    rms_mean,entropy_mean = get_stats(x,sr)
                    
                    """
                    these values were determined by estimation over a large
                    number of files. See check_thresholds.py for details. Note
                    that the 'or' can be changed to 'and' for better quality
                    audio snippets. However, this comes at the cost of
                    increased computation and less output
                    """
                    
                    if rms_mean > 0.035 or entropy_mean < 6.0:
                        
                        num_processed += 1
                        
                        # 1 more window is successfully admitted
                        
                        num_passed += 1
                        
                        # store the metadata
                        
                        data[i].append([path,start,start + window_length])
        
        # convert each dataset to pandas dataframe after processing
        
        data[i] = pd.DataFrame(data[i])
    
    if verbose:
        
        # number of files per class
        
        print('\nNumber of files per class:')
        
        for label in ['0','1']:
            num_files = sum(paths.str[0] == label)
            print('{} class = {} files'.format(label,num_files))
        
        # show train, val, test split ratios
        
        print('\nData split percentages:')
        
        total_size = sum(len(x) for x in data)
        
        for data_split,name in zip(data,['train','val','test']):
            percentage = len(data_split)/total_size*100
            print('{} percentage = {:.2f}%'.format(name,percentage))
        
        # show train, val, test class split ratios to verify stratification
        
        print('\nOriginal class percentages:')
        for label in ['0','1']:
            percentage = sum(paths.str[0] == label)/len(paths)*100
            print('{} class percentage = {:.2f}%'.format(label,percentage))
        
        print('\nNew class split percentages:')
        
        for data_split,name in zip(data,['train','val','test']):
            # other = 0
            # cough = 1
            # speech = 2
            for label in ['0','1']:
                percentage = sum(data_split[0].str[0] == label)/len(data_split)*100
                print('{} {} percentage = {:.2f}%'.format(label,name,percentage))
    
    # write the training, validation, and testing data to csv
    
    print('\nWriting csv files...')
    print('-'*40)
    
    data[0].to_csv(str(dst_dir / 'data_train.csv'),
                      header = False,
                      index = False)
    
    data[1].to_csv(str(dst_dir / 'data_val.csv'),
                    header = False,
                    index = False)
    
    data[2].to_csv(str(dst_dir / 'data_test.csv'),
                     header = False,
                     index = False)
