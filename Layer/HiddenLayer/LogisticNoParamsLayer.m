classdef LogisticNoParamsLayer < NoParamsHiddenLayer
   
   properties
      isLocallyLinear = false
   end
   
   methods
      function [y, ffExtras] = feed_forward(~, x)
         y = 1./(1+exp(-x));
         ffExtras = [];
      end   
      
      function value = compute_Dy(~, x, y)
         u = exp(-x)./(1 + exp(-x)); % u = 1-y
         value = y.*u;
      end
      
      function value = compute_D2y(~, ~, y, Dy)
         value = Dy.*(1-2*y);
      end
   end
end

