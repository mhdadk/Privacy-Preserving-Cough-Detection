import numpy as np

# this is an example audio signal. It was sampled at 4 samples/second.
# 0's followed by 1's followed by 0's represents a cough

x = np.array([0,0,0,0,1,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,0])

# example 0,1 predictions by classifier for each overlapping window

preds = np.array([0,1,1,0,1,1,1,1,1,1])

# compute the first and last indices of the coughs in the signal using
# edge detection (first-order difference)

edges = np.convolve(x,np.array([1,-1]),'full')
cough_start = np.where(edges == 1)[0]
# need the - 1 since filtered signal is delayed by 1 sample and a full
# convolution is used
cough_end = np.where(edges == -1)[0] - 1

# since the example audio signal was sampled at 4 Hz, then a 1.5 second
# window covers 6 samples

sample_rate = 4 # Hz
window_length = 1.5 # seconds
window_length = int(sample_rate * window_length) # samples
overlap = 2/3 # fraction of window_length

# how many samples to slide the window by to achieve correct overlap

step_size = int(window_length * (1 - overlap))

# threshold used to determine the label for a window based on the maximum
# intersection ratio computed for the window

threshold = 0.5;

# to accumulate prediction statistics and form a confusion matrix

CM = {'TP':0,
      'TN':0,
      'FP':0,
      'FN':0}

"""
for each cough in the window, compute the ratio of the intersection
between the cough and the window, to the total length of the cough,
including the length of the cough inside the window and the length of
the cough outside the window. This is used to determine the label of
the window.

For example, suppose the window looks like this:

...0 1 1 |1 1 0 0| 0 0...

where |...| is the window. Then the ratio is the length of the cough
inside the window, which is 2 samples, to the total length of the
cough, both inside and outside the window, which is 4 samples. Notice
that if the entire cough is inside the window, then the ratio is equal
to 1. Essentially, this ratio is a measure of how much cough is in the
window.

Suppose that there is more than one cough in the window, as shown
below:

...0 0 1 1 |1 1 1 0 0 1 1| 1 1 0 0...

In this case, the ratio is computed for both coughs, and the larger
ratio is chosen as the ratio assigned to the window. In this case, the
ratio for the first cough is 3/5 = 0.6, while the ratio for the second cough
is 2/4 = 1/2 = 0.5. Therefore, the ratio for the first cough is
assigned to the window.

Once a ratio is assigned to a window, a threshold is used to
determine if this ratio is high enough so that the window is labelled
as a cough (1), or otherwise labelled as a non-cough (0). This label is
then compared to the prediction by the classifier to compute prediction
statistics, including true positives, true negatives, false positives,
and false negatives. These statistics are accumulated for each window
for all test signals.

Finally, a precision-recall curve is constructed using these
accumulated statistics, and the average precision is computed from this
precision-recall curve. The window length that maximizes this average
precision is chosen.
"""

# overlapping sliding window

for i,j in enumerate(range(0, x.shape[0] - window_length + 1, step_size)):
    
    # to record the maximum intersection ratio
    
    prev_ratio = 0;
    
    # iterate over each cough to check if it intersects with the window
    
    for cough in zip(cough_start,cough_end):
        
        # compute intersection length
        
        lower = max(cough[0], j)
        upper = min(cough[1], j + window_length - 1)
        
        # if lower > upper, then there is no intersection between the
        # cough and the window, so intersection_length = 0
        
        if lower <= upper:
            intersection_length = upper - lower + 1
        else:
            intersection_length = 0
        
        # compute maximum intersection ratio
        
        ratio = max(intersection_length / (cough[1] - cough[0] + 1),
                    prev_ratio)
        
        # no need to check other coughs if ratio = 1
        
        if ratio == 1.0:
            break
        
        prev_ratio = ratio;
    
    # once the maximum ratio for the window is computed, compare it to a
    # threshold to determine the label for the window
    
    window_label = ratio > threshold
    
    # accumulate prediction statistics
    
    if preds[i] == 1 and window_label == 1: # true positive
        CM['TP'] += 1
    elif preds[i] == 1 and window_label == 0: # false positive
        CM['FP'] += 1
    elif preds[i] == 0 and window_label == 0: # true negative
        CM['TN'] += 1
    elif preds[i] == 0 and window_label == 1: # false negative
        CM['FN'] += 1
