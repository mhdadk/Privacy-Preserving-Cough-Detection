% load cough audio template

filename = '../../data/testing/audioset_3_1.wav';
[template,fs] = audioread(filename);

% convert stereo to mono and make row vector

template = mean(template,2)';

% set to true to listen to cough template

play_original = false;

if play_original
    soundsc(template,fs)
end

% filter then downsample the template by a factor of 2

y = decimate(template,2);

% set to true to listen to decimated cough template

play_ds = true;

if play_ds
    soundsc(y,fs)
end

% plot the original template and the downsampled template

ax1 = subplot(2,1,1);
stem(template,'-')
title('Original template')
xlabel('Sample number')
ax2 = subplot(2,1,2);
stem(y,'-')
title('Decimated template')
xlabel('Sample number')
linkaxes([ax1,ax2])