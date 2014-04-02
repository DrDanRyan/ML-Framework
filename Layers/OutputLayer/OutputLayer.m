classdef OutputLayer < handle
   % Defines the OutputLayer interface (note some methods are provided by
   % the Layer superclass)

   methods (Abstract)
      y = feed_forward(x)
      [grad, dLdx, y] = backprop(obj, x, t)
      loss = compute_loss(obj, y, t)
      init_params(obj)
      increment_params(obj, delta)
      push_to_GPU(obj)
      gather(obj)
   end   
end