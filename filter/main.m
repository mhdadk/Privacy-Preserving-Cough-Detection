template_name = '../../data/testing/zapsplat_1.wav';
[template,Fs] = audioread(template_name);
% stem(template(:,1))
% soundsc(template,Fs)

test_name = '../../data/testing/audioset_3.wav';
[test,Fs] = audioread(test_name);
stem(test(:,1))
soundsc(test,Fs)