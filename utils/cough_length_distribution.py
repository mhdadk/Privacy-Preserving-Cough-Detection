import librosa
import os
import numpy as np

data_dir = '../../data/cough'

files = os.listdir(data_dir)

lengths = np.zeros((len(files),))

for i,file in enumerate(files):
    x,sr = librosa.load(path = os.path.join(data_dir,file),
                        sr = None,
                        mono = True)
    lengths[i] = x.shape[0]/sr

mean = lengths.mean()

variance = lengths.var(ddof=1)