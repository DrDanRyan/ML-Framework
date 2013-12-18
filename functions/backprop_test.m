function [grad_errors, sens_error] = backprop_test(layer, x)
eps = 1e-3;

% Use backprop method
y = layer.feed_forward(x, true);
dLdy = gpuArray.ones(size(y));
[grad, dLdx] = layer.backprop(x, y, dLdy);

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
      FD_grad{p}(i) = sum(dydp(:))/size(x, 2);
      layer.params{p}(i) = layer.params{p}(i) + eps; % return to original value
   end
end

% Finite difference sensetivity (dLdx)
FD_dLdx = gpuArray.nan(size(dLdx));
for i = 1:length(x)
   x(i) = x(i) + eps;
   posVal = layer.feed_forward(x);
   x(i) = x(i) - 2*eps;
   negVal = layer.feed_forward(x);
   dydx = (posVal - negVal)/(2*eps);
   FD_dLdx(i) = sum(dydx(:));
   x(i) = x(i) + eps;
end

% Compute differences
grad_errors = cellfun(@(bp, fd) gather(max(abs(bp(:) - fd(:)))), grad, FD_grad, 'UniformOutput', false);
sens_error = gather(max(abs(dLdx(:) - FD_dLdx(:))));
end

