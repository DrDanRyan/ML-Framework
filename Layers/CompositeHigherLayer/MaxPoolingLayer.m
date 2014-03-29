classdef MaxPoolingLayer < matlab.mixin.Copyable
   
   properties
      dydx
   end
   
   methods
      function y = feed_forward(obj, x, isSave)
         y = max(x, [], 3);
         if nargin == 3 && isSave
            obj.dydx = bsxfun(@eq, x, y);
         end
      end
      
      function dLdx = backprop(obj, dLdy)
         dLdx = bsxfun(@times, obj.dydx, dLdy);
         obj.dydx = [];
      end
   end
   
end

