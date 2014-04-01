classdef TanhLayer < CompositeHigherLayer & matlab.mixin.Copyable
   % Applies an elementwise tanh to the input.
   
   properties
      dydx
   end

   methods
      function y = feed_forward(obj, x, isSave)
         v = exp(-2*x);
         u = 2./(1 + v);
         y = u - 1; % robust tanh(x)
         
         if nargin == 3 && isSave
            obj.dydx = v.*u.*u;
         end
      end   
      
      function dLdx = backprop(obj, dLdy)
         dLdx = obj.dydx.*dLdy;
         obj.dydx = [];
      end
      
   end
end

