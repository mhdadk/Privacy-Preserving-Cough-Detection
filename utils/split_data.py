import os
from sklearn.model_selection import train_test_split
import pandas as pd

data_dir = '../../datasets/2'
data_split_dir = '../../data_split'

paths = []

for i,label in enumerate(os.listdir(data_dir)):
    for file in os.listdir(os.path.join(data_dir,label)):
        paths.append([os.path.join(label,file),i])

data = pd.DataFrame(paths)

rng_seed = 42

data_train,data_val = train_test_split(data,
                                       train_size = 0.8,
                                       random_state = rng_seed,
                                       shuffle = True,
                                       stratify = data.iloc[:,1])

data_val,data_test = train_test_split(data_val,
                                      train_size = 0.5,
                                      random_state = rng_seed,
                                      shuffle = True,
                                      stratify = data_val.iloc[:,1])

# show training, validation, and testing class ratios

for data,name in zip([data_train,data_val,data_test],
                     ['train','val','test']):
    for label in os.listdir(data_dir):
        print('{} {} ratio = {}'.format(label,name,
                                        sum(data.iloc[:,1] == i)/len(data)))
    
# write to csv

data_train.to_csv(os.path.join(data_split_dir,'data_train.csv'),
                  header = False,
                  index = False)

data_val.to_csv(os.path.join(data_split_dir,'data_val.csv'),
                header = False,
                index = False)

data_test.to_csv(os.path.join(data_split_dir,'data_test.csv'),
                 header = False,
                 index = False)
