classdef SoftplusLayer < matlab.mixin.Copyable
   
   properties
      C   % y = log(1 + exp(C*x))/C ~ max(0, x) (larger C leads to closer approx)
      dydx
   end
   
   methods
      function obj = SoftplusLayer(C)
         if nargin < 1
            C = 5;
         end
         obj.C = C;
      end
      
      function y = feed_forward(obj, x, isSave)
         if nargin == 3 && isSave
            expCX = exp(obj.C*x);
            one_plus_expCX = 1 + expCX;
            y = log(one_plus_expCX)/obj.C;
            obj.dydx = expCX./one_plus_expCX;
         else
            y = log(1 + exp(obj.C*x))/obj.C;
         end
         
         % replace degenerate values
         yInf = y == Inf;
         if any(yInf(:))
            y(yInf) = x(yInf);
         end
      end
      
      function dLdx = backprop(obj, dLdy)
         dLdx = obj.dydx.*dLdy;
         obj.dydx = [];
      end
      
   end
end

