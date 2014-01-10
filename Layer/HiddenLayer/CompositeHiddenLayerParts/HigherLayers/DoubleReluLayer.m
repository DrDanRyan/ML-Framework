classdef DoubleReluLayer < matlab.mixin.Copyable
   % A double rectified linear unit (rectified at x = 0 and x = 1)
   
   properties
      dydx
   end
   
   methods
      function y = feed_forward(obj, x, isSave)
         y = min(1, max(0, x));
         
         if nargin == 3 && isSave
            obj.dydx = x > 0 & x < 1;
         end
      end
      
      function dLdx = backprop(obj, dLdy)
         dLdx = obj.dydx.*dLdy;
         obj.dydx = [];
      end
      
   end
end

