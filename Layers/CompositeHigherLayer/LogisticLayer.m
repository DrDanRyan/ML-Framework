classdef LogisticLayer < matlab.mixin.Copyable
   
   properties
      dydx
   end
   
   methods
      function y = feed_forward(obj, x, isSave)
         y = 1./(1+exp(-x));
         if nargin == 3 && isSave
            obj.dydx = exp(-x).*y.*y;
         end
      end   
      
      function dLdx = backprop(obj, dLdy)
         dLdx = obj.dydx.*dLdy;
         obj.dydx = [];
      end
      
   end
end

