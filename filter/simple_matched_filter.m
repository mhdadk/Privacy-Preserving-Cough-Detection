% load sample audio

filename = '../../data/testing/audioset_3.wav';
[sample,~] = audioread(filename);

% convert stereo to mono and make row vector

sample = mean(sample,2)';

% load cough audio template

filename = '../../data/testing/audioset_3_1.wav';
[template,~] = audioread(filename);

% convert stereo to mono and make row vector

template = mean(template,2)';

% time-reverse template to make matched filter

template = fliplr(template);

% filter sample audio with time-reversed template

output = filter(template,1,sample);

% plot sample audio and result of matched filtering

subplot(2,1,1)
stem(sample)
title('Sample Audio')
xlabel('Sample number')
subplot(2,1,2)
stem(output)
title('Filtered signal')
xlabel('Sample number')