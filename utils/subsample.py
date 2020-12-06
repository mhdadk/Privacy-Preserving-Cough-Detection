import os
import random
import shutil

src_dir = '../../datasets/1'
dst_dir = '../../datasets/8'
num_cough_files = 431

for label in os.listdir(src_dir):
    if label == 'COUGH':
        continue
    files = os.listdir(os.path.join(src_dir,label))
    subsample = random.sample(files,num_cough_files)
    for file in subsample:
        shutil.copy2(os.path.join(src_dir,label,file),
                     os.path.join(dst_dir,label,file))
        