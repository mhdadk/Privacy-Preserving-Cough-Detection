% get list of audio files

files = dir('../../data/cough_clean');

% new sampling rate

Fs_new = 16000;

% turn warning for audio clipping off

warning('off','MATLAB:audiovideo:audiowrite:dataClipped')

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
    
    dst = strcat('../../data/cough_clean_16kHz/',files(i).name);
    audiowrite(dst,x_new,Fs_new)
    
end
