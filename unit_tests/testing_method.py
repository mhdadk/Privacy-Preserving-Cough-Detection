import torch

"""
Suppose a signal is 6 samples long and sampled at 2 samples per second.
This means that:

0.5 seconds = 1 sample
1 second = 2 samples
1.5 seconds = 3 samples
2 seconds = 4 samples
2.5 seconds = 5 samples
3 seconds = 6 samples

Suppose that this signal is:

1,0,1,0,1,0

Where 1 is a sample that corresponds to a cough, and 0 is a sample that
does not correspond to a cough.

Suppose that for cough detection, we use a window that is
0.5 seconds (1 sample) long. Then, suppose that the classifier predicted:
    
0,0,1,1,1,0

This means that the classifier had an accuracy of 4/6 = 2/3 = 66.7% for
a window with a length of 0.5 seconds.

Now suppose that a 1 second (2 samples) long window is used instead. Then,
the classifier could predict:
    
0,1,0

It is difficult to compare the performance of this classifier with the first
classifier. However, to better compare them, these predictions can be upsampled
using nearest neighbor as follows:
    
0,1,0 --> 0,0,1,1,0,0

The classifier then has an accuracy of 2/3 or 66.7%. Relating this to
actual coughs, since coughs have a maximum length of around 1 second,
then the 1 second window will be the basis for nearest neighbor upsampling.

NEED TO CONSIDER WINDOW OVERLAP

Suppose 

"""
x = torch.tensor([1,0,1,0,1,0])
