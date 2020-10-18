import os
import librosa
import sounddevice as sd

wd = '../../../data_archive/data_audio/1_cough'
file = 'fsd_16.wav'

x,sr = librosa.load(path = os.path.join(wd,file),
                    offset = 0.0,
                    duration = 0.41956916099773245,
                    sr = None)

sd.play(x,sr)
