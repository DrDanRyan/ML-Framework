function data = linear_normalize(data)
% Linear transformation to make data have zero mean and unit variance.

data = bsxfun(@minus, data, mean(data, 2)); % subtract row means
data = bsxfun(@rdivide, data, std(data, [], 2)); % divide by row std
end

