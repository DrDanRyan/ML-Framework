% All base layers should implement the method:
% y = feed_forward(obj, x, isSave)
% objCopy = copy(obj)

% If the base layer has params, it should implement:
% [grad, dLdx] = backprop(obj, x, dLdy)
% as well as increment_params, init_params, gather, push_to_GPU

% If the base layer does not have params, then backprop should only 
% return dLdx.