import numpy as np
from sklearn import metrics

# this is the long signal that contains coughs

x = np.random.randn(16)

# this is a binary mask that represents where coughs are located in the
# long signal. This is similar to a bounding box in images. Consecutive
# 0's followed by consecutive 1's followed by consecutive 0's indicates a
# single cough

y = np.array([0,0,1,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0])

# suppose that a 3 sample window with 0% overlap is used to analyze the
# original signal x. This means that there would be 6 labels for each
# 18 sample signal. However, how would a window such as [1,1,0] be
# labelled? This is where a threshold is needed and a mean average
# precision metric is helpful. Initially, suppose that at least 50% of the
# window should contain 1's so that the entire window is labelled as a
# cough. This means that the labels for each window in x are:

y_windows = np.array([0,1,0,1,1,1])

# now suppose that the classifier predicted the following sequence for
# each non-overlapping window

y_hat = np.array([0,1,1,1,0,1])

# next, the precision and recall for these predictions can be computed

precision1 = metrics.precision_score(y_windows,y_hat)
recall1 = metrics.recall_score(y_windows,y_hat)

# however, note that these precision and recall values are a function of
# the threshold that was chosen such that a window is labelled as a cough
# or not cough. This threshold was set to 50%. Suppose that this threshold
# is now set to 25%. This means that the labels for x now become

y_windows = np.array([1,1,1,1,1,1])

# and the new precision and recall values are

precision2 = metrics.precision_score(y_windows,y_hat)
recall2 = metrics.recall_score(y_windows,y_hat)

#%% this process can be repeated for several thresholds

thresholds = np.linspace(0,1,int(1/0.2))
precisions = []
recalls = []

for threshold in thresholds:
    y_windows = []
    # generate y_windows for each threshold
    for i in range(0,len(y)-1,3):
        window = y[i:i+3]
        y_windows.append(window.mean() > threshold)
    # convert to np array
    y_windows = np.array(y_windows)
    # compute precision and recall
    precision = metrics.precision_score(y_windows,y_hat)
    precisions.append(precision)
    recall = metrics.recall_score(y_windows,y_hat,zero_division=1)
    recalls.append(recall)
    
# next, the area under the precision-recall (AUPR) curve can be computed.
# This is also known as the average precision (AP). It is computed using
# trapezoidal integration

AUPR = np.trapz(precisions,recalls)

"""
However, the problem with using a non-overlapping sliding window is that
the step size will be large, and so it will be difficult to detect coughs
with different lengths. Therefore, 
"""