function [grad_errors, sens_error] = backprop_test(layer, x, dLdy)
eps = 1e-3;
if nargin < 2
   x = gpuArray.rand(layer.inputSize, 1);
   dLdy = gpuArray.rand(layer.outputSize, 1);
end

% Use backprop method
[y, ffExtras] = layer.feed_forward(x);
[grad, dLdx] = layer.backprop(x, y, ffExtras, dLdy);

% Finite difference gradients
FD_grad = cell(size(grad));
for p = 1:length(layer.params)
   FD_grad{p} = gpuArray.nan(size(layer.params{p}));
   for i = 1:length(layer.params{p})
      layer.params{p}(i) = layer.params{p}(i) + eps;
      posVal = layer.feed_forward(x);
      layer.params{p}(i) = layer.params{p}(i) - 2*eps;
      negVal = layer.feed_forward(x);
      dydp = (posVal - negVal)/(2*eps);
      FD_grad{p}(i) = sum(dLdy(:).*dydp(:));
      layer.params{p}(i) = layer.params{p}(i) + eps; % return to original value
   end
end

% Finite difference sensetivity (dLdx)
FD_dLdx = nan(size(dLdx));
for i = 1:length(x)
   x(i) = x(i) + eps;
   posVal = layer.feed_forward(x);
   x(i) = x(i) - 2*eps;
   negVal = layer.feed_forward(x);
   dydx = (posVal - negVal)/(2*eps);
   FD_dLdx(i) = gather(sum(dLdy(:).*dydx(:)));
   x(i) = x(i) + eps;
end

% Compute differences
grad_errors = cellfun(@(bp, fd) gather(max(abs(bp(:) - fd(:)))), grad, FD_grad);
sens_error = gather(max(abs(dLdx(:) - FD_dLdx(:))));
end

