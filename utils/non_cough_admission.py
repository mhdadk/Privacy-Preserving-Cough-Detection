import numpy as np
import librosa

x,sr = librosa.load(path = '../../data/cough/zapsplat_1.wav',
                    sr = None,
                    mono = True)

if len(x.shape)

# frames are 64 ms wide

frame_width = int(0.064 * sr)

num_frames = int(np.floor())

