import os
from sklearn.model_selection import train_test_split
import csv

cough_dir = '../../data/cough'
speech_dir = '../../data/speech'
other_dir = '../../data/other'

coughs = os.listdir(cough_dir)
speech = os.listdir(speech_dir)
other = os.listdir(other_dir)

rng_seed = 42

coughs_train,coughs_val = train_test_split(coughs,
                                           train_size = 0.8,
                                           random_state = rng_seed,
                                           shuffle = True,
                                           stratify = None)

coughs_val,coughs_test = train_test_split(coughs_val,
                                          train_size = 0.5,
                                          random_state = rng_seed,
                                          shuffle = True,
                                          stratify = None)

print('coughs_train ratio = {}'.format(len(coughs_train)/len(coughs)))
print('coughs_val ratio = {}'.format(len(coughs_val)/len(coughs)))
print('coughs_test ratio = {}'.format(len(coughs_test)/len(coughs)))

with open('../../data_split/cough_train.csv',mode='w') as file:
    writer = csv.writer(file,
                        delimiter=',',
                        lineterminator = '\n',
                        quoting=csv.QUOTE_MINIMAL)
    for name in coughs_train:
        writer.writerow(['cough/'+name])
        
with open('../../data_split/cough_val.csv',mode='w') as file:
    writer = csv.writer(file,
                        delimiter=',',
                        lineterminator = '\n',
                        quoting=csv.QUOTE_MINIMAL)
    for name in coughs_val:
        writer.writerow(['cough/'+name])

with open('../../data_split/cough_test.csv',mode='w') as file:
    writer = csv.writer(file,
                        delimiter=',',
                        lineterminator = '\n',
                        quoting=csv.QUOTE_MINIMAL)
    for name in coughs_test:
        writer.writerow(['cough/'+name])

speech_train,speech_val = train_test_split(speech,
                                           train_size = 0.8,
                                           random_state = rng_seed,
                                           shuffle = True,
                                           stratify = None)

speech_val,speech_test = train_test_split(speech_val,
                                          train_size = 0.5,
                                          random_state = rng_seed,
                                          shuffle = True,
                                          stratify = None)

print('speech_train ratio = {}'.format(len(speech_train)/len(speech)))
print('speech_val ratio = {}'.format(len(speech_val)/len(speech)))
print('speech_test ratio = {}'.format(len(speech_test)/len(speech)))

with open('../../data_split/speech_train.csv',mode='w') as file:
    writer = csv.writer(file,
                        delimiter=',',
                        lineterminator = '\n',
                        quoting=csv.QUOTE_MINIMAL)
    for name in speech_train:
        writer.writerow(['speech/'+name])
        
with open('../../data_split/speech_val.csv',mode='w') as file:
    writer = csv.writer(file,
                        delimiter=',',
                        lineterminator = '\n',
                        quoting=csv.QUOTE_MINIMAL)
    for name in speech_val:
        writer.writerow(['speech/'+name])

with open('../../data_split/speech_test.csv',mode='w') as file:
    writer = csv.writer(file,
                        delimiter=',',
                        lineterminator = '\n',
                        quoting=csv.QUOTE_MINIMAL)
    for name in speech_test:
        writer.writerow(['speech/'+name])

other_train,other_val = train_test_split(other,
                                         train_size = 0.8,
                                         random_state = rng_seed,
                                         shuffle = True,
                                         stratify = None)

other_val,other_test = train_test_split(other_val,
                                        train_size = 0.5,
                                        random_state = rng_seed,
                                        shuffle = True,
                                        stratify = None)

print('other_train ratio = {}'.format(len(other_train)/len(other)))
print('other_val ratio = {}'.format(len(other_val)/len(other)))
print('other_test ratio = {}'.format(len(other_test)/len(other)))

with open('../../data_split/other_train.csv',mode='w') as file:
    writer = csv.writer(file,
                        delimiter=',',
                        lineterminator = '\n',
                        quoting=csv.QUOTE_MINIMAL)
    for name in other_train:
        writer.writerow(['other/'+name])
        
with open('../../data_split/other_val.csv',mode='w') as file:
    writer = csv.writer(file,
                        delimiter=',',
                        lineterminator = '\n',
                        quoting=csv.QUOTE_MINIMAL)
    for name in other_val:
        writer.writerow(['other/'+name])

with open('../../data_split/other_test.csv',mode='w') as file:
    writer = csv.writer(file,
                        delimiter=',',
                        lineterminator = '\n',
                        quoting=csv.QUOTE_MINIMAL)
    for name in other_test:
        writer.writerow(['other/'+name])
