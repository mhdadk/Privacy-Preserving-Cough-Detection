import numpy as np

# this is the long signal that contains coughs. Consecutive 0's followed by
# consecutive 1's followed by consecutive 0's indicates a single cough.
# Note also that this is a binary mask that will also be used for actual
# signals

x = np.array([0,0,1,1,1,0,0,0,1,1,1,0,0])