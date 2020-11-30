import librosa
import os
import numpy as np

data_dir = '../../temp_data/cough_long'

files = os.listdir(data_dir)

lengths = np.zeros((len(files),))

short_files = []

for i,file in enumerate(files):
    lengths[i] = librosa.get_duration(filename = os.path.join(data_dir,file))
    if lengths[i] < 1.5:
        short_files.append((file,lengths[i]))
        # os.remove(os.path.join(data_dir,file))

mean = lengths.mean()

variance = lengths.var(ddof=1)