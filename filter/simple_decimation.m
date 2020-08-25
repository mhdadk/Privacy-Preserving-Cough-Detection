% load cough audio template

filename = '../../data/testing/audioset_3_1.wav';
[template,~] = audioread(filename);

% convert stereo to mono and make row vector

template = mean(template,2)';

% set to true to listen to cough template

play_original = true;

if play_original
    soundsc(template)
end

% filter then downsample the template by a factor of 2

y = decimate(template,2);

% set to true to listen to downsampled cough template

play_ds = true;

if play_ds
    soundsc(y)
end

% plot the original template and the downsampled template

subplot(2,1,1)
stem(template,LineSpec,'-')
title('Sample Audio')
xlabel('Sample number')
subplot(2,1,2)
stem(output,LineSpec,'-')
title('Filtered signal')
xlabel('Sample number')