import os
from sklearn.model_selection import train_test_split
import pandas as pd

datasets_dir = '../../datasets'
datasets_splits_dir = '../../datasets_splits'

# 1,2,3,4,5,6, or 7
for dataset in os.listdir(datasets_dir):
    # skip if not directory or if the '1' or '2' directories
    if (dataset == '1' or 
        dataset == '2' or
        not os.path.isdir(os.path.join(datasets_dir,dataset))):
        continue
    # to store paths and labels for each dataset
    paths = []
    # COUGH, ESC50, FSDKAGGLE2018, LIBRISPEECH, or RESP
    for i,label in enumerate(os.listdir(os.path.join(datasets_dir,dataset))):
        for file in os.listdir(os.path.join(datasets_dir,dataset,label)):
            paths.append([os.path.join(label,file),label[0]])
    
    data = pd.DataFrame(paths)

    rng_seed = 42
    
    data_train,data_val = train_test_split(data,
                                           train_size = 0.8,
                                           random_state = rng_seed,
                                           shuffle = True,
                                           stratify = data.iloc[:,1])
    
    # ensure that ESC50 data is in training dataset
    
    for i,file in data_val.iterrows(index=False,name=None):
    
    data_val,data_test = train_test_split(data_val,
                                          train_size = 0.5,
                                          random_state = rng_seed,
                                          shuffle = True,
                                          stratify = data_val.iloc[:,1])
    
    # show train, val, test split ratios
    
    print('-'*50)
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
    
# write to csv

# data_train.to_csv(os.path.join(data_split_dir,'data_train.csv'),
#                   header = False,
#                   index = False)

# data_val.to_csv(os.path.join(data_split_dir,'data_val.csv'),
#                 header = False,
#                 index = False)

# data_test.to_csv(os.path.join(data_split_dir,'data_test.csv'),
#                  header = False,
#                  index = False)
