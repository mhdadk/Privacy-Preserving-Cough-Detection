import librosa
import os
import numpy as np

data_dir2 = '../../data/other_cropped'

files2 = os.listdir(data_dir2)

lengths2 = np.zeros((len(files2),))

for i,file in enumerate(files2):
    lengths2[i] = librosa.get_duration(filename = os.path.join(data_dir2,file))

# mean = lengths.mean()
# variance = lengths.var(ddof=1)