% this is an example audio signal. It was sampled at 4 samples/second.
% 0's followed by 1's followed by 0's represents a cough

x = [0 0 0 0 1 1 1 1 0 0 0 1 1 1 0 0 0 1 1 1 1 1 0 0];

% example 0,1 predictions by classifier for each overlapping window

preds = [0,1,1,0,1,1,1,1,1,1];

% get the first and last indices of the coughs in the signal

[cough_start,cough_end] = compute_cough_ind(x);

% since the example audio signal was sampled at 4 Hz, then a 1.5 second
% window covers 6 samples

sample_rate = 4; % Hz
window_length = 1.5; % seconds
window_length = sample_rate * window_length; % samples
overlap = 2/3; % fraction of window_length

% how many samples to slide the window by to achieve correct overlap

step_size = window_length * (1 - overlap);

% to index the sliding window

window_num = 0;

% threshold used to determine the label for a window based on the maximum
% ratio computed for the window

threshold = 0.5;

% to accumulate prediction statistics and form a confusion matrix

CM = struct('TP',0,...
            'TN',0,...
            'FP',0,...
            'FN',0);

%{
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
%}
        
% overlapping sliding window

for i = 1 : step_size : length(x) - window_length + 1
    window_num = window_num + 1;
    % to record the maximum intersection ratio
    prev_ratio = 0;
    % find the coughs that intersect with the window
    cough_idx = find(cough_end >= i & ...
                     cough_start <= i + window_length - 1);
    % iterate over the coughs that intersect with the window
    for cough = [cough_start(cough_idx) ; cough_end(cough_idx)]
        % compute intersection length
        lower = max(cough(1),i);
        upper = min(cough(2),i + window_length - 1);
        % guaranteed that upper >= lower
        intersection_length = upper - lower + 1;
        % compute maximum intersection ratio
        ratio = max(intersection_length / (cough(2) - cough(1) + 1),...
                    prev_ratio);
        prev_ratio = ratio;
    end
    
    % once the maximum ratio for the window is computed, compare it to a
    % threshold to determine the label for the window
    
    window_label = ratio > threshold;
    
    % accumulate prediction statistics
    
    if preds(window_num) == 1 && window_label == 1 % true positive
        CM.TP = CM.TP + 1;
    elseif preds(window_num) == 1 && window_label == 0 % false positive
        CM.FP = CM.FP + 1;
    elseif preds(window_num) == 0 && window_label == 0 % true negative
        CM.TN = CM.TN + 1;
    elseif preds(window_num) == 0 && window_label == 1 % false negative
        CM.FN = CM.FN + 1;
    end
end

% get the first and last indices of the coughs in the signal using edge
% detection (first-order difference)

function [cough_start,cough_end] = compute_cough_ind(signal)
    locs = conv(signal,[1,-1],'full');
    cough_start = find(locs == 1);
    % need the - 1 since filtered signal is delayed by 1 sample and a full
    % convolution is used
    cough_end = find(locs == -1) - 1;
end