classdef Model < handle 
   % This defines the Model interface
   
   methods (Abstract)
      grad = gradient(obj, x, t)
      y = output(obj, x)
      loss = compute_loss(obj, x, t)
      increment_params(obj, delta)
      gather(obj)
      push_to_GPU(obj)
      objCopy = copy(obj)
      reset(obj)
   end
   
end

