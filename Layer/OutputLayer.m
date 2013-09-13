classdef OutputLayer < matlab.mixin.Copyable
   % Defines the OutputLayer interface (note some methods are provided by
   % the Layer superclass)
   
   methods (Abstract)
      [grad, dLdx, output] = backprop(obj, x, t)
      loss = compute_loss(obj, y, t)
      y = feed_forward(x)
      push_to_GPU(obj)
      gather(obj)
      increment_params(obj, delta_params)
      init_params(obj)
   end
   
end