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

# where the raw data is located

data_dir = pathlib.Path('../../data/raw_test')

# new sample rate to resample audio to in Hz

new_sr = 16000

# how long extracted windows should be in seconds

window_lengths = [1.0, 1.5, 2.0, 2.5, 3.0]

# number of windows to extract per cough

num_windows = 20

# whether to show train, val, and test ratios and class split ratios

verbose = True

# turn off SettingWithCopyWarning. default is 'warn'. Just in case,
# this should be set to 'warn' during testing

pd.options.mode.chained_assignment = None

# initialize random number generator

rng = np.random.default_rng()

for window_length in window_lengths:
    
    print('\nProcessing windows of length {} seconds'.format(window_length))
    
    # to store the processing instructions for the data
    
    data = []
    
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
    else:
        dst_dir.mkdir()
    
    # csv file containing timestamps of coughs

    fp_cough_ts = open('cough_timestamps.csv')
    
    # need to reset the generator every time it is iterated over
    
    cough_ts_reader = csv.reader(fp_cough_ts, delimiter=',')
    
    for dataset in data_dir.iterdir():
        
        print('\nDataset: {}'.format(dataset))
        
        # to count the number of processed files

        num_processed = 0
        
        """
        if analyzing the 1_COUGH dataset, then extract <num_windows> windows
        of length <window_length> seconds that contain the coughs 
        """
        
        if 'COUGH' in str(dataset):
            
            for row in cough_ts_reader:
                
                # convert path to pathlib.Path object for easier handling
                
                row[0] = pathlib.Path(row[0])
            
                # get the length of the audio file in seconds
                
                split_filename = str(row[0]).split('_')
                # TODO for testing
                if (int(split_filename[1]) > 50 or
                    'esc' in split_filename[0] or
                    'fsd' in split_filename[0]):
                    continue
                file = dataset / (split_filename[0]+'_'+split_filename[1]+row[0].suffix)
                file_length = librosa.get_duration(filename = str(file))
                
                # compute the length of the cough in seconds
                
                cough_length = float(row[2]) - float(row[1])
                
                """
                if the entire audio file is not at least as long as 1.5
                times the length of the window to be extracted, skip the
                audio file. Also, if the length of the cough is greater
                than the length of the window to be extracted, skip this
                cough
                """
                
                if (file_length < 1.5 * window_length or
                    cough_length > window_length):
                    continue
                
                # track progress
                
                num_processed += 1                
                print('\rProcessed {} files'.format(num_processed),
                      end='',flush=True)                
                
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
                """
                
                low = max(0,float(row[2]) - window_length)
                high = float(row[1])
                
                # extract <num_windows> windows around this cough
                
                for _ in range(num_windows):
                    
                    # sample a random starting time for the window
                    
                    start = rng.uniform(low = low, high = high)
                    
                    # store the metadata
                    
                    path = pathlib.Path(dataset.name,file.name).as_posix()
                    data.append([path,new_sr,start,start + window_length])
                        
            fp_cough_ts.close()
        
        else:
            
            for file in dataset.iterdir():
                
                # get the length of the audio file in seconds
                
                file_length = librosa.get_duration(filename = str(file))
            
                # if the entire audio file is not at least as long as 1.5
                # times the length of the window to be extracted, skip the
                # audio file
            
                # skip files that are shorter than snippet_length
            
                if file_length < 1.5 * window_length:
                    continue
                
                # track progress
                
                num_processed += 1                
                print('\rProcessed {} files'.format(num_processed),
                      end='',flush=True)
                
                # number of windows that have been admitted
            
                num_passed = 0
            
                """
                sometimes, only 1 or 2 short sounds exist in a long audio file.
                In this case, stop searching after <iter_count> iterations has
                passed
                """
                
                iter_count = 0
            
                """
                start randomly searching for snippets until <max_passed> 
                windows have successfully been admitted
                """
                
                max_passed = 5
                
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
                    
                    x,sr = librosa.load(path = str(file),
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
                        
                        # 1 more window is successfully admitted
                        
                        num_passed += 1
                        
                        # store the metadata
                        
                        path = pathlib.Path(dataset.name,file.name).as_posix()
                        data.append([path,new_sr,start,start + window_length])
    
    print('\n\nSplitting data...')            
    
    data = pd.DataFrame(data)
    
    rng_seed = 42
    
    data_train,data_val = train_test_split(data,
                                           train_size = 0.8,
                                           random_state = rng_seed,
                                           shuffle = True,
                                           stratify = data[0].str[0].astype(int))
    
    """
    ensure that ESC50 and RESP data is in training dataset and not
    validation or testing.
    
    This is done by first iterating through data_val, checking if each file
    is an ESC50 or RESP file. If not, the file is skipped. Otherwise,
    randomly sample a file from data_train to be switched with the ESC50
    or RESP file in data_val. If the file that was sampled from data_train
    is already an ESC50 or RESP file, then sample again. Repeat this until
    the randomly sampled file is no longer a ESC50 or RESP file. However, 
    to maintain the desired class split ratios, the sampled file cannot be
    a COUGH or LIBRISPEECH file either.
    
    Therefore, the sampled file must be a FSDKAGGLE2018 file. However,
    if there are no FSDKAGGLE2018 files in data_train, then switching
    the ESC50 or RESP file in data_val with a file in data_train will not
    be possible. That is what the if statement is for.
    
    Additionally, it is possible that there are more ESC50 or RESP files
    in data_val than there are FSDKAGGLE2018 files in data_train. In this
    case, there will be ESC50 or RESP files leftover in data_val. However,
    since there are more FSDKAGGLE2018 files than both ESC50 and RESP files
    combined, then this will not be a problem    
    """
    #%%
    # if data_train contains fsd files
    if data_train[0].str.contains('fsd').any():
        # find the indices of fsd files in data_train
        train_idx = data_train[0][data_train[0].str.contains('fsd')].index
        # find the indices of esc and resp files in data_val
        val_idx = data_val[0][data_val[0].str.contains('esc|resp')].index
        # iterate over the esc and resp file indices in data_val. For this
        # to work properly, len(train_idx) >= len(val_idx) must be true
        if len(train_idx) >= len(val_idx):
            for src_idx in val_idx:
                # if not an ESC50 file or not a RESP file, skip
                # if 'esc' not in row[1] and 'resp' not in row[1]:
                #     continue
                # sample a random index of an fsd file in data_train
                sample = rng.integers(0,len(train_idx))
                dst_idx = train_idx[sample]
                # remove the sampled index so that it is not sampled again
                train_idx = train_idx.delete(sample)
                # # repeated sampling
                # while 'fsd' not in data_train.loc[dst_idx][0]:
                #     dst_idx = np.random.choice(data_train.index)
                # switch the files
                temp = data_val.loc[src_idx].copy()
                data_val.loc[src_idx] = data_train.loc[dst_idx].copy()
                data_train.loc[dst_idx] = temp
                # if train_idx is empty after sampling all indices, break
                if len(train_idx) == 0:
                    break
        else:
            print('\nCould not remove all ESC50 and RESP files from ' +
                  'validation and testing data.')
    
    if verbose:
        # check that data_val no longer contains esc files
        print('\ndata_val contains ESC50 files? {}'.format(data_val[0].str.contains('esc').any()))
        print('data_val contains RESP files? {}'.format(data_val[0].str.contains('resp').any()))
    
    data_val,data_test = train_test_split(data_val,
                                          train_size = 0.5,
                                          random_state = rng_seed,
                                          shuffle = True,
                                          stratify = data_val[0].str[0].astype(int))
    
    if verbose:
        
        # number of files per class
        
        print('\nNumber of files per class:')
        
        for label in ['0','1','2']:
            num_files = sum(data[0].str[0] == label)
            print('{} class = {} files'.format(label,num_files))
        
        # show train, val, test split ratios
        
        print('\nData split percentages:')
        
        for data_split,name in zip([data_train,data_val,data_test],
                                   ['train','val','test']):
            percentage = len(data_split)/len(data)*100
            print('{} percentage = {:.2f}%'.format(name,percentage))
        
        # show train, val, test class split ratios to verify stratification
        
        print('\nOriginal class percentages:')
        for label in ['0','1','2']:
            percentage = sum(data[0].str[0] == label)/len(data)*100
            print('{} class percentage = {:.2f}%'.format(label,percentage))
        
        print('\nNew class split percentages:')
        
        for data_split,name in zip([data_train,data_val,data_test],
                                   ['train','val','test']):
            # other = 0
            # cough = 1
            # speech = 2
            for label in ['0','1','2']:
                percentage = sum(data_split[0].str[0] == label)/len(data_split)*100
                print('{} {} percentage = {:.2f}%'.format(label,name,percentage))
    
    # write the training, validation, and testing data to csv
    
    print('\nWriting csv files...')
    print('-'*40)
    
    data_train.to_csv(str(dst_dir / 'data_train.csv'),
                      header = False,
                      index = False)
    
    data_val.to_csv(str(dst_dir / 'data_val.csv'),
                    header = False,
                    index = False)
    
    data_test.to_csv(str(dst_dir / 'data_test.csv'),
                     header = False,
                     index = False)
