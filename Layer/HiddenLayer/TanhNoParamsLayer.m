classdef TanhNoParamsLayer < NoParamsHiddenLayer
   
   properties
      isLocallyLinear = false
   end
   
   methods
      function [y, ffExtras] = feed_forward(~, x)
         v = exp(-2*x);
         u = 2./(1 + v);
         y = u - 1; % robust tanh(x)
         ffExtras = [];
      end   
      
      function value = compute_Dy(~, x, y)
         v = exp(-2*x);
         u = 2./(1 + v);
         value = v.*u.*u;
      end
      
      function value = compute_D2y(~, ~, y, Dy)
         value = -2*y.*Dy;
      end
   end
end

