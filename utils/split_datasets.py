import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

datasets_dir = '../../datasets'
datasets_splits_dir = '../../datasets_splits'
# whether to show train, val, and test ratios and class split ratios
verbose = True
# turn off SettingWithCopyWarning. default is 'warn'. Just in case,
# this should be set to 'warn' during testing
pd.options.mode.chained_assignment = None

# 1,2,3,4,5,6, or 7
for dataset in os.listdir(datasets_dir):
    # skip if not directory or if the '1' or '2' directories
    if (dataset == '1' or 
        dataset == '2' or
        not os.path.isdir(os.path.join(datasets_dir,dataset))):
        continue
    # to store paths and labels for each dataset
    paths = []
    # 1_COUGH, 0_ESC50, 0_FSDKAGGLE2018, 2_LIBRISPEECH, or 0_RESP
    for label in os.listdir(os.path.join(datasets_dir,dataset)):
        for file in os.listdir(os.path.join(datasets_dir,dataset,label)):
            paths.append([os.path.join(label,file),label[0]])
    
    data = pd.DataFrame(paths)

    rng_seed = 42
    
    data_train,data_val = train_test_split(data,
                                           train_size = 0.8,
                                           random_state = rng_seed,
                                           shuffle = True,
                                           stratify = data.iloc[:,1])
    
    # ensure that ESC50 and RESP data is in training dataset and not
    # validation or testing
    
    for src_idx,filename,_ in data_val.itertuples(index=True,name=None):
        # if not an ESC50 file or not a RESP file, skip
        if 'esc' not in filename and 'resp' not in filename:
            continue
        # sample a random index from the training data
        dst_idx = np.random.choice(data_train.index)
        # keep sampling to make sure that we are only switching ESC50 and
        # RESP files in the val or test datasets with FSDKAGGLE2018 files in
        # the training dataset to not affect the class split ratios
        while 'FSD' not in data_train.loc[dst_idx][0]:
            dst_idx = np.random.choice(data_train.index)
        # once finished sampling, switch the files
        temp = data_val.loc[src_idx].copy()
        data_val.loc[src_idx] = data_train.loc[dst_idx].copy()
        data_train.loc[dst_idx] = temp
    
    if verbose:
        # check that data_val no longer contains esc files
        print('-'*50)
        print('data_val contains ESC50 files? {}'.format(data_val[0].str.contains('esc').any()))
        print('data_val contains RESP files? {}'.format(data_val[0].str.contains('resp').any()))
    
    data_val,data_test = train_test_split(data_val,
                                          train_size = 0.5,
                                          random_state = rng_seed,
                                          shuffle = True,
                                          stratify = data_val.iloc[:,1])
    
    if verbose:
        
        # show train, val, test split ratios
        
        print('\nData split ratios for dataset {}:'.format(dataset))
        
        for data_split,name in zip([data_train,data_val,data_test],
                                   ['train','val','test']):
            print('{} ratio = {}'.format(name,len(data_split)/len(data)))
        
        # show train, val, test class split ratios to verify stratification
        
        print('\nOriginal class split ratios for dataset {}:'.format(dataset))
        for label in ['0','1','2']:
            print('{} class ratio = {}'.format(label,
                                               sum(data.iloc[:,1] == label)/len(data)))
        
        print('\nNew class split ratios for dataset {}:'.format(dataset))
        
        for data_split,name in zip([data_train,data_val,data_test],
                                   ['train','val','test']):
            # other = 0
            # cough = 1
            # speech = 2
            for label in ['0','1','2']:
                print('{} {} ratio = {}'.format(label,name,
                                                sum(data_split.iloc[:,1] == label)/len(data_split)))
    
    # create the folders 3,4,5,6, and 7 if they do not already exist
    
    if not os.path.isdir(os.path.join(datasets_splits_dir,dataset)):
        os.mkdir(os.path.join(datasets_splits_dir,dataset))
    
    # write to csv
    
    data_train.to_csv(os.path.join(datasets_splits_dir,dataset,'data_train.csv'),
                      header = False,
                      index = False)
    
    data_val.to_csv(os.path.join(datasets_splits_dir,dataset,'data_val.csv'),
                    header = False,
                    index = False)
    
    data_test.to_csv(os.path.join(datasets_splits_dir,dataset,'data_test.csv'),
                      header = False,
                      index = False)
