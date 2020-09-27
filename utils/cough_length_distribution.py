import librosa
import os

data_dir = '../../data/cough'

for file in os.listdir(data_dir):
    x,sr = librosa.load(path = os.path.join(data_dir,file),
                        sr = None)
    
