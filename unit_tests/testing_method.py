import torch

"""
Suppose that the following signal is sampled at 1 sample per second:
    
1,0,1,1,0,1,1,0,1

Where 1 is a sample or set of samples that correspond to a cough, and
0 is a sample or set of samples that don't correspont to a cough.

This means that window sizes have lengths of:

1 second = 1 sample
2 seconds = 2 samples
...

Suppose that for cough detection, a 1 second window with 0% overlap is
used. This means that a possible sequence of predictions given the signal
above is:
    
0,0,1,0,1,1,1,1,1

Notice that the sequence of predictions has the same length as the input
signal, since each window corresponds to a single sample. This sequence
of predictions means that the classifier achieved an accuracy of
approximately 56%.

Now suppose that for cough detection, a 2 second window with 50% overlap is
used. This means that a possible sequence of predictions given the signal
above is:
    
1,0,1,0,1,0,1,0

Notice that this sequence of predictions is 1 sample shorter than the input
signal. Because the input signal has 9 samples and the sequence of
predictions has 8 samples, then this sequence of predictions can be
interpolated using nearest-neighbor or linear interpolation. In the case
of linear interpolation, the resulting filtered sequence can be rounded to
provide binary predictions.

Another method is to filter then downsample the original input signal to
match the length of the sequence of predictions.




    
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
