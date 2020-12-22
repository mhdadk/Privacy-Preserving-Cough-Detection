import numpy as np
from sklearn import metrics

# this is an example audio signal. It was sampled at 4 samples/second.
# 0's followed by 1's followed by 0's represents a cough

x = np.array([0,0,0,0,1,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,0])

# compute the first and last indices of the coughs in the signal using
# edge detection (first-order difference)

def get_cough_ind(signal):
    edges = np.convolve(signal,np.array([1,-1]),'full')
    cough_start = np.where(edges == 1)[0]
    # need the - 1 since filtered signal is delayed by 1 sample and a full
    # convolution is used
    cough_end = np.where(edges == -1)[0] - 1
    return cough_start,cough_end

cough_start,cough_end = get_cough_ind(x)
