% get list of cough audio files

files = dir('../../data/cough_audio_clean_aug');

% create augmenter object

augmenter = audioDataAugmenter(...
    "AugmentationMode","sequential", ...
    "AugmentationParameterSource","random",...
    "NumAugmentations",200, ...
    ...
    "TimeStretchProbability",0.7, ...
    "SpeedupFactorRange", [0.6,1.7], ...
    ...
    "PitchShiftProbability",0, ...
    ...
    "VolumeControlProbability",0.5, ...
    "VolumeGainRange",[0,10], ... % in dB
    ...
    "AddNoiseProbability",0.4, ...
    "SNRRange",[0,1], ...
    ...
    "TimeShiftProbability",0);

% start at 3 to skip '.' and '..' files

for i = 3:length(files)
    
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
        audiowrite(new_filename,audio_file,Fs);
    end
    
end

% augment audio

% x_aug = augment(augmenter,x,Fs);

% for i = 1:height(x_aug)
%     disp(x_aug.AugmentationInfo(i))
% end
% for i=1:height(x_aug)
%     audiowrite('.../test/zapsplate_1')
% end