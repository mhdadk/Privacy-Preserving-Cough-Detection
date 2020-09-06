% get list of cough audio files

files = dir('../../data/cough_audio_clean_aug');

% create augmenter object

augmenter = audioDataAugmenter(...
    "AugmentationMode","sequential", ...
    "AugmentationParameterSource","random",...
    "NumAugmentations",200, ...
    ...
    "TimeStretchProbability",0.8, ...
    "SpeedupFactorRange", [0.6,1.7], ...
    ...
    "PitchShiftProbability",0, ...
    ...
    "VolumeControlProbability",0.5, ...
    "VolumeGainRange",[0,10], ... % in dB
    ...
    "AddNoiseProbability",0.4, ...
    "SNRRange",[10,20], ...
    ...
    "TimeShiftProbability",0);

% turn warning for audio clipping off

warning('off','MATLAB:audiovideo:audiowrite:dataClipped')

% start at 3 to skip '.' and '..' files

for i = 3:length(files)
    
    % progress update
    
    fprintf('Augmenting %s... (%d/%d)\n',files(i).name,i-2,length(files)-2);
    
    % create absolute filename
    
    filename = fullfile(files(i).folder,files(i).name);
    
    % load audio file
    
    [x,Fs] = audioread(filename);
    
    % perform augmentation
    
    x_aug = augment(augmenter,x,Fs);
    
    % save each augmented file
    
    for j = 1:height(x_aug)
        audio_file = x_aug.Audio{j};
        idx = strcat('_',int2str(j),'.wav');
        new_filename = strrep(files(i).name,'.wav',idx);
        new_full_filename = fullfile(files(i).folder,new_filename);
        audiowrite(new_full_filename,audio_file,Fs);
    end
    
end
