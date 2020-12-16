import numpy as np

"""
suppose we have a 1.5 second window of audio sampled at 12 Hz that
contains several coughs. An example of this is shown below. 0's followed
by 1's followed by 0's represents a cough. This is a binary mask, similar
to a bounding box in images
"""

y = np.array([0,0,1,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0])

# suppose that a classifier classifies this 1.5 second window as a cough.
# This means that every sample in the window is classified as a cough,
# as shown below

y_hat = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

# whether this is an accurate detection or not depends on the intersection
# over union (IoU) ratio

IoU = np.mean(y == y_hat)

# next, a threshold should be chosen for whether this detection was correct
# or not. A 50% threshold i