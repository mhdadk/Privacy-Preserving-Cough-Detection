import pathlib
import numpy as np

data_dir = pathlib.Path('../../data/raw')

num_files_per_label = []
for label in data_dir.iterdir():
    num_files_per_label.append(len(list(label.iterdir())))
num_files_per_label = np.array(num_files_per_label)

label_probs = num_files_per_label / sum(num_files_per_label)

rng = np.random.default_rng(seed = 42)

"""
Let X be a multinomial random variable representing the training,
validation, or testing dataset, such that the range of X is {0,1,2}.
Also, let Y be a multinomial random variable representing the 0_AUDIOSET,
0_ESC50, 0_FSDKAGGLE2018, 0_RESP, 1_COUGH, 2_LIBRISPEECH datasets, such
that the range of Y is {0,1,2,3,4,5}. Since it is required that the
proportions of each label be the same in the training, validation, and
testing datasets, then it is required that:
    
p(Y = y|X = x) = p(Y = y)

This is only true if X and Y are independent. This also means that:
    
p(X = x|Y = y) = p(X = x)

Therefore, to ensure that the proportions of labels in the training,
validation, and testing datasets is the same as the proportions of labels
in the entire dataset, it is only required to sample a label from p(Y = y),
sample, without replacement, a file from the folder associated with that
label, sample, with replacement, the training, validation, or
testing dataset from p(X = x), and finally assign the sampled file to that
dataset.
"""

# simulate the files in their folders

files = []

for i,num_files in enumerate(num_files_per_label):
    files.append([i] * num_files)

files = [[0]*num_files_per_label[0],
         [1]]

# sample several labels to simulate data splitting

Y = rng.multinomial(n = 1,
                    pvals = label_probs,
                    size = sum(num_files_per_label)).argmax(axis=1)


