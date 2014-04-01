classdef MaxNormalizeLayer < CompositeHigherLayer & matlab.mixin.Copyable
   % divides all features by max of absolute value of features
   
   properties
      maxVals
   end
   
   methods
      function y = feed_forward(obj, x, isSave)
         maxVals = max(abs(x)); %#ok<*PROP>
         maxVals(maxVals == 0) = 1e-4;
         y = bsxfun(@rdivide, x, maxVals);
         
         if nargin == 3 && isSave
            obj.maxVals = maxVals;
         end
      end
      
      function dLdx = backprop(obj, dLdy)
         dLdx = bsxfun(@rdivide, dLdy, obj.maxVals);
         obj.maxVals = [];
      end
      
   end
end

