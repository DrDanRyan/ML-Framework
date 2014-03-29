classdef CompositeHigherLayer < handle
   % This defines the interface for CompositeHigherLayers

   methods (Abstract)
      y = feed_forward(obj, x, isSave)
      objCopy = copy(obj)
      
      % Note: the signature for this function does not use x or y.
      % If the CompositeHigherLayer does not have any parameters that need to be
      % learned, then backprop should only return dLdx.
      [grad, dLdx] = backprop(obj, dLdy)
      
      % If the CompositeHigherLayer does have parameters to learn, it needs to
      % implement the following methods (these are not necessary if no
      % parameters are present):
      % increment_params(obj)
      % init_params(obj)
      % gather(obj)
      % push_to_GPU(obj)
   end
   
end