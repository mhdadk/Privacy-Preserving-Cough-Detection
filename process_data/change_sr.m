% get list of audio files

files = dir('../../data/data_audio/0_not_cough');

% new sampling rate

Fs_new = 16000;

% iterate through files

for i = 3:length(files)
    
    % progress update
    
    fprintf('Processing %s... (%d/%d)\n',files(i).name,i-2,length(files)-2);
    
    % create absolute filename
    
    filename = fullfile(files(i).folder,files(i).name);
    
    % load audio file
    
    [x_old,Fs_old] = audioread(filename);
    
    % compute resampling ratio
    
    [p,q] = rat(Fs_new/Fs_old);
    
    % decimate signal
    
    x_new = resample(x_old,p,q);
    
    % save downsampled audio file
    
    dst = strcat('../../data/data_audio/not_cough_16kHz/',files(i).name);
    audiowrite(dst,x_new,Fs_new)
    
end
