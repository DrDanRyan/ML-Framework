classdef SoftplusNoParamsLayer < matlab.mixin.Copyable
   
   properties
      dydx
   end
   
   methods
      function y = feed_forward(obj, x, isSave)
         if nargin == 3 && isSave
            expX = exp(x);
            one_plus_expX = 1 + expX;
            y = log(one_plus_expX);
            obj.dydx = max(expX./one_plus_expX, 1e-14);
         else
            y = log(1 + exp(x));
         end
         
         % replace degenerate values
         yInf = y == Inf;
         yZero = y == 0;
         if any(yInf(:))
            y(yInf) = x(yInf);
         end
         
         if any(yZero(:))
            y(yZero) = 1e-14;
         end
      end
      
      function dLdx = backprop(obj, dLdy)
         dLdx = obj.dydx.*dLdy;
         obj.dydx = [];
      end
      
   end
end

