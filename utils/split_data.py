import os
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from glob import glob

data_dir = '../../temp_data/short'
data_split_dir = '../../data_split'

paths = []

for i,label1 in enumerate(os.listdir(data_dir)):
    for j,label2 in enumerate(os.listdir(os.path.join(data_dir,label1))):
        for file in os.listdir(os.path.join(data_dir,label1,label2)):
            paths.append([os.path.join(label1,file),i])
    
data = pd.DataFrame(paths)

rng_seed = 42

X_train,X_val,y_train,y_val = train_test_split(data.iloc[:,0], # paths
                                               data.iloc[:,1], # labels
                                               train_size = 0.8,
                                               random_state = rng_seed,
                                               shuffle = True,
                                               stratify = data.iloc[:,1])





    # filenames = glob(os.path.join(data_dir,label,'*'))
    
    # # for i in range(len(filenames)):
    # #     filenames[i] = filenames[i].replace('\\','/')
    
    # rng_seed = 42
    
    # train,val = train_test_split(filenames,
    #                              train_size = 0.8,
    #                              random_state = rng_seed,
    #                              shuffle = True,
    #                              stratify = None)

    # val,test = train_test_split(val,
    #                             train_size = 0.5,
    #                             random_state = rng_seed,
    #                             shuffle = True,
    #                             stratify = None)
    
    # print('{} train ratio = {}'.format(label,len(train)/len(filenames)))
    # print('{} val ratio = {}'.format(label,len(val)/len(filenames)))
    # print('{} test ratio = {}'.format(label,len(test)/len(filenames)))
    
    # with open(os.path.join(data_split_dir,label+'_train.pkl'),'wb') as fp:
    #     pickle.dump(train,fp)
    
    # with open(os.path.join(data_split_dir,label+'_val.pkl'),'wb') as fp:
    #     pickle.dump(val,fp)
    
    # with open(os.path.join(data_split_dir,label+'_test.pkl'),'wb') as fp:
    #     pickle.dump(test,fp)
    