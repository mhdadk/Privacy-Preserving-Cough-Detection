import numpy as np

# this is the long signal that contains coughs

x = np.random.randn(15)

# this is a binary mask that represents where coughs are located in the
# long signal. This is similar to a bounding box in images. Consecutive
# 0's followed by consecutive 1's followed by consecutive 0's indicates a
# single cough

y = np.array([0,0,1,1,1,0,0,0,1,1,1,0,0,0,0])

# suppose that a 2 sample window with 50% (1 sample) overlap is used to
# analyze the original signal x. This means that in total, there would be
# 14 overlapping windows of x. These are shown below in x_windows.

windows = []

window_size = 2
step_size = int(0.5 * window_size)

for i in range(0,len(x)-1,step_size):
    windows.append(x[i:i+window_size])

x_windows = np.array(windows)

# suppose that predictions are made using these overlapping windows for
# non-cough (0) and cough (1) instances. An example of what this could be
# is shown in y_hat

y_hat = np.array([0,1,1,1,1,0,0,1,1,1,1,0,0,0])

# next, labels need to be assigned to these overlapping windows using the
# y array. Note that this step can be combined with the previous step for
# efficiency

labels = []

for i in range(0,len(y)-1,step_size):
    labels.append(y[i:i+window_size])
    
y_windows = np.array(labels)