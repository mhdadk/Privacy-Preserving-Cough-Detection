import torch
import torchaudio
import matplotlib.pyplot as plt
import sounddevice as sd

# sr = 16000
# length_sec = 1.5
# x = torch.rand((int(sr*length_sec),))

x,sr = torchaudio.load(filepath = '../snippets/4/2_LIBRISPEECH/ls_1_1.wav')

win_length_sec = 0.012
win_length = int(sr * win_length_sec)
# need 50% overlap to satisfy constant-overlap-add constraint to allow
# for perfect reconstruction using inverse STFT
hop_length = int(sr * win_length_sec / 2)

spec = torchaudio.transforms.Spectrogram(n_fft = 512,
                                         win_length = win_length,
                                         hop_length = hop_length,
                                         pad = 0,
                                         window_fn = torch.hann_window,
                                         power = None,
                                         normalized = False,
                                         wkwargs = None)

y = spec(x)

mag,phase = torchaudio.functional.magphase(y)

# assume we reconstructed the magnitude spectrogram

# z = |z| cos(phase) + j |z| * sin(phase)

real = torch.multiply(mag,torch.cos(phase))
imag = torch.multiply(mag,torch.sin(phase))

y_hat = torch.stack((real,imag),dim = -1)

y_diff = torch.mean(torch.abs(y - y_hat))

# perform inverse STFT

x_hat = torch.istft(y_hat,
                    n_fft = 512,
                    hop_length = hop_length,
                    win_length = win_length,
                    window = torch.hann_window(win_length),
                    center = True,
                    normalized = False,
                    onesided = None,
                    length = None,
                    return_complex = False)

x_diff = torch.mean(torch.abs(x - x_hat))

#%% test reconstructed audio

plt.subplot(2,1,1)
plt.plot(x[0])
plt.subplot(2,1,2)
plt.plot(x_hat[0])

#%%

sd.play(x[0],sr)

#%%

sd.play(x_hat[0],sr)