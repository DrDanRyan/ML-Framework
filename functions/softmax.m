function y = softmax(x)

y = bsxfun(@minus, x, max(x, 1)); % rescale incoming activations so max is 0
y = exp(y);
y = bsxfun(@rdivide, y, sum(y, 1)); % rescale outputs so they sum to 1

end

