import os
import numpy as np
import librosa
import csv

def get_start_and_end(x,template):
    
    # used to ignore correlations involving large values in x
    
    threshold = np.correlate(template,template,'valid')[0]
    
    # cross-correlate template with signal
    
    signal = np.correlate(x,template,'same')
    
    # set values that are too large to 0
    
    signal[signal>threshold] = 0
    
    # find the center of the template in the signal
    
    center = np.argmax(signal)
    
    # if length of template is odd
    
    template_length = template.shape[0]
    
    if template_length % 2:
        
        start = center - int(np.floor(template_length/2))
        end = center + int(np.floor(template_length/2))
        
    else: # if length of template is even
        
        start = center - int(template_length/2)
        end = center + int(template_length/2) - 1
    
    return start,end

short_cough_dir = '../../data/cough'
long_cough_dir = '../../data_archive/data_audio/1_cough'

files = os.listdir(short_cough_dir)

# open the csv file for writing

fp = open('../../timestamps.csv', mode='w')

# initialize csv writer

csv_writer = csv.writer(fp,
                        delimiter = ',',
                        lineterminator = '\n',
                        quotechar = '"',
                        quoting = csv.QUOTE_MINIMAL)

for file in files:
    
    temp = file.split('_')[:2]
    long_file = temp[0]+'_'+temp[1]+'.wav'

    x,sr = librosa.load(path = os.path.join(long_cough_dir,long_file),
                        sr = None)

    template = librosa.load(path = os.path.join(short_cough_dir,file),
                            sr = None)[0]
    
    # get first and last sample numbers of template
    
    start,end = get_start_and_end(x,template)
    
    # write starting and ending times of template in seconds
    
    csv_writer.writerow([long_file,start/sr,end/sr])

# close the csv file

fp.close()