classdef TRecLayer < CompositeHigherLayer & matlab.mixin.Copyable
   % Applies an elementwise truncated relu to the input. Any input with value
   % less than theta is mapped to zero.
   
   properties
      theta % cutoff value
      dydx
   end
   
   methods
      function obj = TRecLayer(theta)
         if nargin < 1
            theta = 1;
         end
         obj. theta = theta;
      end
      
      function y = feed_forward(obj, x, isSave)
         y = x;
         cutIdx = y < obj.theta;
         y(cutIdx) = 0;
         
         if nargin == 3 && isSave
            obj.dydx = ~cutIdx;
         end
      end
      
      function dLdx = backprop(obj, dLdy)
         dLdx = dLdy.*obj.dydx;
         obj.dydx = [];
      end
      
   end
end

