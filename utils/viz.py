import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

cough_path = '../../data_archive/cough_clean/zapsplat_1.wav'
speech_path = '../../test.wav'

cough,sr = librosa.load(path = cough_path,
                        sr = None,
                        mono = True)

mel_spec = librosa.feature.melspectrogram(y = cough,
                                          sr = sr,
                                          n_fft = 4096,
                                          hop_length = 480)

S_dB = librosa.power_to_db(mel_spec, ref=np.max)
img = librosa.display.specshow(S_dB,
                                x_axis='time',
                                y_axis='mel',
                                sr=sr,
                                fmax=8000)
plt.xlabel('Time (s)',fontsize=18)
plt.ylabel('Frequency (Hz)',fontsize=18)
plt.colorbar(format='%+2.0f dB')

# speech,_ = librosa.load(path = speech_path,
#                         sr = None,
#                         mono = True)

# plt.subplot(2,1,1)
# t = np.arange(cough.shape[0])/sr
# plt.plot(t,cough)
# locs, labels = plt.xticks()
# plt.xticks(ticks=locs,labels=labels,fontsize=15)
# locs, labels = plt.yticks()
# plt.yticks(ticks=locs,labels=labels,fontsize=15)
# plt.title('Cough Signal',fontsize=18)
# plt.xlabel('Time (s)',fontsize=18)
# plt.ylabel('Amplitude',fontsize=18)
# plt.subplot(2,1,2)
# plt.title('Speech signal of someone saying "it was"')
# plt.plot(speech)
