function error_rate = binary_error(y, t)

if any(t < 0) % svm targets
   threshold = 0;
   t(t < 0) = 0; % change to {0, 1} targets
else % logistic targets
   threshold = .5;
end

predictions = y > threshold;
error_rate = sum(predictions ~= t)/size(y, 2);
end

