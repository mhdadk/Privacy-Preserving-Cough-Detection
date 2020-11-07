import csv
import soundfile as sf
import librosa
import os
import numpy as np

# NOTE: audioset_77_3.wav and fsd_169_5.wav are longer than 1 second,
# so they were removed from the csv file

long_cough_dir = '../../../data_archive/data_audio/1_cough'
dst_dir = '../../../data_archive/cough_1-5_seconds'
sample_rate = 16000

# lengths = np.zeros((1118,))

fp = open('cough_timestamps.csv')
csv_reader = csv.reader(fp,delimiter=',')

snippet_length = 1.5

for row in csv_reader:
    temp = row[0].split('_')[:2]
    filename = temp[0]+'_'+temp[1]+'.wav'
    file_length = librosa.get_duration(filename = os.path.join(long_cough_dir,filename))
    
    # skip audio files that are less than snippet_length seconsd long
        
    if file_length < snippet_length:
        continue
    
    cough_length = float(row[2]) - float(row[1])
    offset = np.random.uniform(low = 0.0,
                               high = snippet_length - cough_length)
    
    # need max function to account for negative starting values
    
    start = max(0,float(row[1]) - offset)
    
    x,sr = librosa.load(path = os.path.join(long_cough_dir,filename),
                        sr = sample_rate,
                        offset = start,
                        duration = snippet_length)
    
    sf.write(file = os.path.join(dst_dir,row[0]),
             data = x,
             samplerate = sr)
    
fp.close()
