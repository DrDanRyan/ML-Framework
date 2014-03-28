classdef HiddenLayer < handle
   % Defines the HiddenLayer interface 

   methods (Abstract)
      
      % If isSave = true, the HiddenLayer must save any relevant information it
      % needs to perform a subsequent backprop pass. If isSave is false, it
      % does not need to prepare for a backprop pass. Default: isSave = false
      y = feed_forward(obj, x, isSave)
      
      % When backprop is called, HiddenLayer is free to clear any stored data
      % that it was saving for the backprop pass. There should be a call to
      % feed_forward(x, true) before the next backprop pass.
      [grad, dLdx] = backprop(obj, x, y, dLdy)
      
      init_params(obj)
      increment_params(obj, delta)
      push_to_GPU(obj)
      gather(obj)
      objCopy = copy(obj)
   end
end

