classdef OutputLayer < handle
<<<<<<< HEAD
   % Defines the OutputLayer interface.
   
=======
   % Defines the OutputLayer interface (note some methods are provided by
   % the Layer superclass)

>>>>>>> e540396056d196ecb0b1b604eb014cdcccd43daf
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