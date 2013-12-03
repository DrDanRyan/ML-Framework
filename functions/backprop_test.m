function grad_errors = backprop_test(layer, x, dLdy)
eps = 1e-3;
if nargin < 2
   x = gpuArray.rand(layer.inputSize, 1);
   dLdy = gpuArray.rand(layer.outputSize, 1);
end

% Use backprop method
[y, ffExtras] = layer.feed_forward(x);
grad = layer.backprop(x, y, ffExtras, dLdy);

% Finite Differences
FD_grad = cell(size(grad));
for p = 1:length(layer.params)
   FD_grad{p} = gpuArray.nan(size(layer.params{p}));
   for i = 1:length(layer.params{p})
      layer.params{p}(i) = layer.params{p}(i) + eps;
      y = layer.feed_forward(x);
      posVal = sum(dLdy(:).*y(:));
      layer.params{p}(i) = layer.params{p}(i) - 2*eps;
      y = layer.feed_forward(x);
      negVal = sum(dLdy(:).*y(:));
      FD_grad{p}(i) = squeeze((posVal - negVal)/(2*eps));
      layer.params{p}(i) = layer.params{p}(i) + eps; % return to original value
   end
end

% Compute differences
grad_errors = cellfun(@(bp, fd) gather(max(abs(bp(:) - fd(:)))), grad, FD_grad);
end

