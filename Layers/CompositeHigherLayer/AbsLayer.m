classdef AbsLayer < CompositeHigherLayer & matlab.mixin.Copyable
   % This layer applies an elementwise absolute value function to the input.
   
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

