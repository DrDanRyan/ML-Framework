function error_rate = multiclass_error(y, t)
% Compute the error rate for multinomial targets and softmax model outputs

[~, yIdx] = max(y);
[~, tIdx] = max(t);
N = size(y, 2);
error_rate = sum(yIdx ~= tIdx)/N;
end

