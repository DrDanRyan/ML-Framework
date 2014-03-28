function error_rate = binary_error(y, t)
% Computes binary error rate for predictions y and targets t. 
% 
% t = targets vector: should be either array of 0's and 1's or -1's and 1's
%
% y = model outputs: will be thresholded at .5 if targets are {0, 1} and
% thresholded at 0 if targets are {-1, 1} to obtain binary predictions

if any(t < 0) % svm targets
   threshold = 0;
   t(t < 0) = 0; % change to {0, 1} targets
else % logistic targets
   threshold = .5;
end

predictions = y > threshold;
error_rate = sum(predictions ~= t)/size(y, 2);
end

