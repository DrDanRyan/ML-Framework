function data = normalize(data)
data = bsxfun(@minus, data, mean(data, 2)); % subtract row means
data = bsxfun(@rdivide, data, std(data, [], 2)); % divide by row std
end

