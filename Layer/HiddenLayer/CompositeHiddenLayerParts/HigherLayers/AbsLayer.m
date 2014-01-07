classdef AbsLayer < matlab.mixin.Copyable
   
   properties
      dydx
   end
   
   methods
      function y = feed_forward(obj, x, isSave)
         y = abs(x);
         
         if nargin == 3 && isSave
            obj.dydx = sign(x);
         end
      end
      
      function dLdx = backprop(obj, dLdy)
         dLdx = obj.dydx.*dLdy;
         obj.dydx = [];
      end
   end
end

