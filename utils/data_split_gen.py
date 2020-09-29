import os
from sklearn.model_selection import train_test_split
import csv

data_dir = '../../data'

for label in os.listdir(data_dir):
    
    filenames = os.listdir(os.path.join(data_dir,label))
    
    rng_seed = 42
    
    train,val = train_test_split(filenames,
                                 train_size = 0.8,
                                 random_state = rng_seed,
                                 shuffle = True,
                                 stratify = None)

    val,test = train_test_split(val,
                                train_size = 0.5,
                                random_state = rng_seed,
                                shuffle = True,
                                stratify = None)
    
    print('{} train ratio = {}'.format(label,len(train)/len(filenames)))
    print('{} val ratio = {}'.format(label,len(val)/len(filenames)))
    print('{} test ratio = {}'.format(label,len(test)/len(filenames)))
    
    data_split_dir = '../../data_split'
    
    if not os.path.isdir(data_split_dir):
        os.mkdir(data_split_dir)
    
    with open(os.path.join(data_split_dir,label+'_train.csv'),mode='w') as fp:
        writer = csv.writer(fp,
                            delimiter=',',
                            lineterminator = '\n',
                            quoting=csv.QUOTE_MINIMAL)
        for filename in train:
            writer.writerow([os.path.join(data_dir,label,filename)])
        
    with open(os.path.join(data_split_dir,label+'_val.csv'),mode='w') as fp:
        writer = csv.writer(fp,
                            delimiter=',',
                            lineterminator = '\n',
                            quoting=csv.QUOTE_MINIMAL)
        for filename in val:
            writer.writerow([os.path.join(data_dir,label,filename)])
    
    with open(os.path.join(data_split_dir,label+'_test.csv'),mode='w') as fp:
        writer = csv.writer(fp,
                            delimiter=',',
                            lineterminator = '\n',
                            quoting=csv.QUOTE_MINIMAL)
        for filename in test:
            writer.writerow([os.path.join(data_dir,label,filename)])