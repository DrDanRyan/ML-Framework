classdef CompositeBaseLayer < handle
   % This defines the CompositeBaseLayer interface
   
   methods (Abstract)
      y = feed_forward(obj, x, isSave)
      
      % This signature does not have y as an input (different from HiddenLayer)
      [grad, dLdx] = backprop(obj, x, dLdy) 
      
      increment_params(obj)
      init_params(obj)
      gather(obj)
      push_to_GPU(obj)
      objCopy = copy(obj)
   end
end