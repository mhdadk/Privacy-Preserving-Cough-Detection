import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import librosa

src_dir = '../../datasets/1'
dst_dir = '../../datasets_splits/8'
# whether to show train, val, and test ratios and class split ratios
verbose = True
# turn off SettingWithCopyWarning. default is 'warn'. Just in case,
# this should be set to 'warn' during testing
pd.options.mode.chained_assignment = None

# create the folder if it does not already exist
    
if not os.path.isdir(dst_dir):
    os.mkdir(dst_dir)

paths = []

# subsample each list of files and concatenate them together
# 1_COUGH, 0_ESC50, 0_FSDKAGGLE2018, 2_LIBRISPEECH, or 0_RESP

for label in os.listdir(src_dir):
    # sample all cough files and make sure dataset is balanced by
    # sampling a quarter of each of the non-cough datasets
    if label == '1_COUGH':
        num_files = 431
    else:
        num_files = round(431/4)
    files = os.listdir(os.path.join(src_dir,label))
    subsample = random.sample(files,num_files)
    for file in subsample:
        # if audio file is less than 1 second long, skip it and resample
        # from the list of files. Keep doing this until the file is no
        # longer less than 1 second long
        duration = librosa.get_duration(filename=os.path.join(src_dir,label,file))
        while duration < 1:
            file = random.choice(files)
            duration = librosa.get_duration(filename=os.path.join(src_dir,label,file))
        paths.append([os.path.join(label,file),label[0]])

data = pd.DataFrame(paths)

rng_seed = 42

data_train,data_val = train_test_split(data,
                                       train_size = 0.8,
                                       random_state = rng_seed,
                                       shuffle = True,
                                       stratify = data.iloc[:,1])

# ensure that ESC50 data is in training dataset and not validation
# or testing

for src_idx,filename,_ in data_val.itertuples(index=True,name=None):
    # if not an ESC50 file, skip
    if 'esc' not in filename:
        continue
    # sample a random index from the training data
    dst_idx = np.random.choice(data_train.index)
    # keep sampling to make sure that we are only switching ESC50 files
    # in the val or test datasets with FSDKAGGLE2018 or RESP files in
    # the training dataset to not affect the class split ratios
    while ('fsd' not in data_train.loc[dst_idx][0] and
           'resp' not in data_train.loc[dst_idx][0]):
        dst_idx = np.random.choice(data_train.index)
    # once finished sampling, switch the files
    temp = data_val.loc[src_idx].copy()
    data_val.loc[src_idx] = data_train.loc[dst_idx].copy()
    data_train.loc[dst_idx] = temp

if verbose:
    # check that data_val no longer contains esc files
    print('-'*50)
    print('data_val contains ESC50 files? {}'.format(data_val[0].str.contains('esc').any()))

data_val,data_test = train_test_split(data_val,
                                      train_size = 0.5,
                                      random_state = rng_seed,
                                      shuffle = True,
                                      stratify = data_val.iloc[:,1])

if verbose:
    
    # show train, val, test split ratios
    
    print('\nData split ratios for dataset 8:')
    
    for data_split,name in zip([data_train,data_val,data_test],
                               ['train','val','test']):
        print('{} ratio = {}'.format(name,len(data_split)/len(data)))
    
    # show train, val, test class split ratios to verify stratification
    
    print('\nOriginal class split ratios for dataset 8:')
    for label in ['0','1','2']:
        print('{} class ratio = {}'.format(label,
                                           sum(data.iloc[:,1] == label)/len(data)))
    
    print('\nNew class split ratios for dataset 8:')
    
    for data_split,name in zip([data_train,data_val,data_test],
                         ['train','val','test']):
        # other = 0
        # cough = 1
        # speech = 2
        for label in ['0','1','2']:
            print('{} {} ratio = {}'.format(label,name,
                                            sum(data_split.iloc[:,1] == label)/len(data_split)))

# write to csv

data_train.to_csv(os.path.join(dst_dir,'data_train.csv'),
                  header = False,
                  index = False)

data_val.to_csv(os.path.join(dst_dir,'data_val.csv'),
                header = False,
                index = False)

data_test.to_csv(os.path.join(dst_dir,'data_test.csv'),
                  header = False,
                  index = False)
    