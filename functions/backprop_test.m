function [grad_errors, sens_error] = backprop_test(layer, x, sensSampleSize)
% Tests a HiddenLayer object to see if its backprop function returns accurate
% values. 
%
% layer = HiddenLayer object
%
% x = input data (can be single feature vec or batch)
%
% sensSampleSize (optional) = integer inidicating sample size for computing dLdx
% accuracy

eps = 1e-4; % small value used for finite difference quotients

% Use backprop method
y = layer.feed_forward(x, true);
dLdy = gpuArray.ones(size(y));
[grad, dLdx] = layer.backprop(x, y, dLdy);

% Create a cell array of deltas for incrementing params in FD calculation
nParams = length(grad);
delta = cell(size(grad));
for idx = 1:length(grad)
   delta{idx} = gpuArray.zeros(size(grad{idx}));
end

% Finite difference gradients
FD_grad = cell(size(grad));
for p = 1:nParams
   FD_grad{p} = gpuArray.nan(size(grad{p}));
   for i = 1:numel(grad{p})
      % Positive perturbation
      delta{p}(i) = eps;
      layer.increment_params(delta);
      posVal = layer.feed_forward(x);
      
      % Negative perturbation
      delta{p}(i) = -2*eps;
      layer.increment_params(delta);
      negVal = layer.feed_forward(x);
      
      % Finite-difference value
      deltaY = posVal - negVal;
      FD_grad{p}(i) = sum(deltaY(:))/(size(x, 2)*2*eps);
      
      % Reset delta to 0 and layer parameters to original value
      delta{p}(i) = eps;
      layer.increment_params(delta);
      delta{p}(i) = 0;
   end
end
grad_errors = cellfun(@(bp, fd) gather(mean(abs(bp(:) - fd(:)))), grad, FD_grad, ...
                        'UniformOutput', false);
clear delta grad


% Finite difference sensetivity (dLdx)
if nargout > 1
   if nargin < 3
      sensSample = 1:numel(x);
   else
      sensSample = randsample(numel(x), sensSampleSize);
   end
   FD_dLdx = gpuArray.nan(size(dLdx));
   for i = sensSample
      x(i) = x(i) + eps;
      posVal = layer.feed_forward(x);
      x(i) = x(i) - 2*eps;
      negVal = layer.feed_forward(x);
      dydx = (posVal - negVal)/(2*eps);
      FD_dLdx(i) = sum(dydx(:));
      x(i) = x(i) + eps;
   end
   sens_error = gather(nanmean(abs(dLdx(:) - FD_dLdx(:))));
end

end

