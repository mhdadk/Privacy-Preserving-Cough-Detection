import librosa
import os
import numpy as np

data_dir = '../../../datasets/7/RESP'

files = os.listdir(data_dir)

lengths = np.zeros((len(files),))

for i,file in enumerate(files):
    lengths[i] = librosa.get_duration(filename = os.path.join(data_dir,
                                                              file))

# mean = lengths.mean()
# variance = lengths.var(ddof=1)