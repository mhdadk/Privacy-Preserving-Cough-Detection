% this is an example audio signal. It was sampled at 4 samples/second.
% 0's followed by 1's followed by 0's represents a cough

% x = [0 0 0 0 1 1 1 1 0 0 0 1 1 1 0 0 0 1 1 1 1 1 0 0 0];
x = [1 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 1 1];

% get the first and last indices of the coughs in the signal using edge
% detection (first-order difference)

[cough_abs_start,cough_abs_end] = get_cough_locs(x);

% since the example audio signal was sampled at 4 Hz, then a 1.5 second
% window covers 6 samples

sample_rate = 4; % Hz
window_length = 1.5; % seconds
window_length = sample_rate * window_length; % samples
overlap = 2/3; % fraction of window_length

% how many samples to slide the window by to achieve correct overlap

step_size = window_length * (1 - overlap);

% number of equally-sized overlapping windows

num_windows = floor((length(x) - window_length) / step_size) + mod(length(x),step_size);

% generate example 0,1 predictions by classifier

p = 0.5;
preds = rand(num_windows,1);
preds = preds < p;

% indices used to slide window

i = 1 : step_size : length(x) - window_length;

% indices used to iterate through predictions

j = 1:length(preds);

% to count the coughs

cough_num = 0;

% drop the last window if it is shorter than all other windows

for idx = [i;j] % iterate through both i and j simultaneously
    
    % get first and last indices of window
    
    window_start = idx(1); % same as i
    window_end = window_start + window_length - 1;
    
    % extract the window
    
    window = x(window_start:window_end);
    
    % get the relative indices of the coughs in the window
    
    [cough_rel_start,cough_rel_end] = get_cough_locs(window);
    
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
    
    for m = [cough_rel_start;cough_rel_end]
        if m(1) == 1 % if cough at start of window
            if m(2) == window_length % if cough covers entire window
                ratio = 1;
                break % no need to check other coughs
            else % if cough does not cover entire window
                % length of cough inside window
                length1 = m(2) - m(1) + 1;
                % length of cough outside window
                cough_start = find(cough_abs_start > window_start - window_length & ...
                                   cough_abs_end >= window_start,1,'last');
                % if cough_start is empty, this means that the cough is
                % completely in the window
                if isempty(cough_start)
                    ratio = 1;
                % otherwise, compute total length of cough inside and outside
                % the window
                else
                    length2 = window_start - cough_abs_start(cough_start);
                    % total length of cough
                    length = length1 + length2;
                
                    ratio = length1 / length;
                end
            end
        % if cough starts somewhere in the window and ends at the end of
        % the window
        elseif m(2) == window_length
            % length of cough inside window
            length1 = m(2) - m(1) + 1;
            % length of cough outside window
            cough_end = find(cough_abs_end < window_end + window_length & ...
                             cough_abs_start <= window_end,1,'first');
            % if cough_end is empty, this means that the cough is
            % completely in the window
            if isempty(cough_end)
                ratio = 1;
            % otherwise, compute total length of cough inside and outside
            % the window
            else
                length2 = cough_abs_end(cough_end) - window_end;
                % total length of cough
                length = length1 + length2;

                ratio = length1 / length;
            end
        % if cough neither starts at the beginning of the window nor ends
        % at the end of the window, then it must be completely in the
        % window, which means that its ratio is 1
        else
            ratio = 1;
        end
    end
end

% get the first and last indices of the coughs in the signal using edge
% detection (first-order difference)

function [cough_start,cough_end] = get_cough_locs(signal)
    locs = conv(signal,[1,-1],'full');
    cough_start = find(locs == 1);
    cough_end = find(locs == -1) - 1;
end