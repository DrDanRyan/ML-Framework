classdef ReluNoParamsLayer < matlab.mixin.Copyable
   
   properties
      dydx
   end
   
   methods
      function y = feed_forward(obj, x, isSave)
         y = max(0, x);
         
         if nargin == 3 && isSave
            obj.dydx = x > 0; % store as logical to reduce memory usage
         end
      end
      
      function dLdx = backprop(obj, dLdy)
         dLdx = obj.dydx.*dLdy;
         obj.Dy = [];
      end
   end
end

