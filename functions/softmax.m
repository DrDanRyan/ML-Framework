function y = softmax(z)
% Fairly robust softmax activation function.

z = bsxfun(@minus, z, max(z, [], 1)); % rescale incoming activations so max is 0
yHat = exp(z);
ySum = sum(yHat, 1);
y = bsxfun(@rdivide, yHat, ySum); % rescale outputs so they sum to 1

end

