import numpy as np

x = np.random.rand(30)
template_start = np.random.randint(low = 0,
                                   high = x.shape[0])
template_end = np.random.randint(low = template_start + 1,
                                 high = x.shape[0])
template = x[template_start:template_end + 1]

threshold = np.correlate(template,template,'valid')[0]

signal = np.correlate(x,template,'same')
signal[signal>threshold] = 0
center = np.argmax(signal)

# if length of template is odd

template_length = template.shape[0]

if template_length % 2:
    
    start = center - int(np.floor(template_length/2))
    end = center + int(np.floor(template_length/2))
    
else: # if length of template is even
    
    start = center - int(template_length/2)
    end = center + int(template_length/2) - 1

if template_start == start and template_end == end:
    print('Well done.')
else:
    print('Try again.')