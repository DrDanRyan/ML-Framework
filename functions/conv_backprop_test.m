function [grad_errors, sens_error] = conv_backprop_test(layer, x, dLdy)
eps = 1e-2;
% Use backprop method
y = layer.feed_forward(x, true);
[grad, dLdx] = layer.backprop(x, y, dLdy);

% Finite difference gradients
FD_grad = cell(size(grad));
for p = 1:length(layer.convLayer.params)
   FD_grad{p} = gpuArray.nan(size(layer.convLayer.params{p}));
   for i = 1:length(layer.convLayer.params{p})
      layer.convLayer.params{p}(i) = layer.convLayer.params{p}(i) + eps;
      posVal = layer.feed_forward(x);
      layer.convLayer.params{p}(i) = layer.convLayer.params{p}(i) - 2*eps;
      negVal = layer.feed_forward(x);
      dydp = (posVal - negVal)/(2*eps);
      FD_grad{p}(i) = sum(dLdy(:).*dydp(:));
      layer.convLayer.params{p}(i) = layer.convLayer.params{p}(i) + eps; % return to original value
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
grad_errors = cellfun(@(bp, fd) gather(max(abs(bp(:) - fd(:)))), grad, FD_grad, 'UniformOutput', false);
sens_error = gather(max(abs(dLdx(:) - FD_dLdx(:))));
end

