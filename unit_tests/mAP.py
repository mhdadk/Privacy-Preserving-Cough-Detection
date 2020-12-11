import numpy as np

# this is the long signal that contains coughs

x = np.random.randn(15)

# this is a binary mask that represents where coughs are located in the
# long signal. This is similar to a bounding box in images. Consecutive
# 0's followed by consecutive 1's followed by consecutive 0's indicates a
# single cough

y = np.array([0,0,1,1,1,0,0,0,1,1,1,0,0,0,0])

