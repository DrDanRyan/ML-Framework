% HigherLayer interface description

% All higher layers should implement:
% y = feed_forward(obj, x, isSave)
% objCopy = copy(obj)

% If there are parameters to be learned in the layer, then 
% they should be stored in the params property (a cell array) and
% backprop should have the form:
% [grad, dLdx] = backprop(obj, dLdy)

% If there are params in the layer, it should also respond to
% increment_params, init_params, gather and push_to_GPU

% If there are no params in the layer, backprop should have the form:
% dLdx = backprop(obj, dLdy)