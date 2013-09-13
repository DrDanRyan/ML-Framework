classdef Model < handle 
   % This defines the Model interface
   
   methods (Abstract)
      [grad, output] = gradient(obj, x, t)
      y = output(obj, x)
      loss = compute_loss(obj, y, t)
      increment_params(obj, delta)
      gather(obj)
      push_to_GPU(obj)
      objCopy = copy(obj)
      reset(obj)
   end
   
end

