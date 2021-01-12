import pathlib
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
Let X be a multinomial random variable representing the training,
validation, or testing dataset, such that the range of X is {0,1,2}.
Also, let Y be a multinomial random variable representing the 0_AUDIOSET,
0_ESC50, 0_FSDKAGGLE2018, 0_RESP, 1_COUGH, 2_LIBRISPEECH datasets, such
that the range of Y is {0,1,2,3,4,5}. Since it is required that the
proportions of each label be the same in the training, validation, and
testing datasets, then it is required that:
    
p(Y = y|X = x) = p(Y = y)

This is only true if X and Y are independent. This also means that:
    
p(X = x|Y = y) = p(X = x)

Therefore, to ensure that the proportions of labels in the training,
validation, and testing datasets is the same as the proportions of labels
in the entire dataset, it is only required to sample a label from p(Y = y),
sample, without replacement, a file from the folder associated with that
label, sample, with replacement, the training, validation, or
testing dataset from p(X = x), and finally assign the sampled file to that
dataset.

Alternatively, if all files are listed together, then the data can be
split by uniformly sampling, without replacement, from this list of files,
sampling, with replacement, the training, validation, or testing dataset
from p(X = x), and finally assign the uniformly sampled file to that
dataset.

This is because, given a discrete uniform random variable W:
    
p(a < W <= b) = (b - a + 1) / |R_W|

Where a,b \in \mathbb{Z} and $R_W$ is the range of W, or the set of all
possible values of W. In this case, $R_W$ is the set of all possible files
in 0_AUDIOSET, 0_ESC50, 0_FSDKAGGLE2018, 0_RESP, 1_COUGH, and 2_LIBRISPEECH
datasets.

Therefore, the probability of each label is proportional to the number of
files that belong to that label.
"""

data_dir = pathlib.Path('../../data/raw')
rng_seed = 42
rng = np.random.default_rng(seed = rng_seed)

# turn off SettingWithCopyWarning. default is 'warn'. Just in case,
# this should be set to 'warn' during testing

pd.options.mode.chained_assignment = None

# simulate data

data = []

for label in data_dir.iterdir():
    for file in label.iterdir():
        data.append('{}/{}'.format(file.parent.name,file.name))

data = pd.DataFrame(data)

# split data

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
            # sample a random index of an fsd file in data_train
            sample = rng.integers(0,len(train_idx))
            dst_idx = train_idx[sample]
            # remove the sampled index so that it is not sampled again
            train_idx = train_idx.delete(sample)
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

# check that data_val no longer contains esc files

print('\ndata_val contains ESC50 files? {}'.format(data_val[0].str.contains('esc').any()))
print('data_val contains RESP files? {}'.format(data_val[0].str.contains('resp').any()))

data_val,data_test = train_test_split(data_val,
                                      train_size = 0.5,
                                      random_state = rng_seed,
                                      shuffle = True,
                                      stratify = data_val[0].str[0].astype(int))
        
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

print('\nAfter augmentation:')

cough_timestamps = {}
fp = open('../utils/cough_timestamps.csv')
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
