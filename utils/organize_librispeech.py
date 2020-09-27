import os
import shutil

data_dir = 'LibriSpeech/train-clean-100'

speakers = os.listdir(data_dir)

for speaker in speakers:
    folders = os.listdir(os.path.join(data_dir,speaker))
    for folder in folders:
        files = os.listdir(os.path.join(data_dir,speaker,folder))
        for file in files:
            if file.endswith('.txt'):
                continue
            src = os.path.join(data_dir,speaker,folder,file)
            dst = os.path.join(data_dir,file)
            shutil.move(src,dst)
