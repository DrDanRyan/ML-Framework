classdef LogisticNoParamsLayer < NoParamsLayer
   
   methods
      function y = feed_forward(obj, x, isSave)
         y = 1./(1+exp(-x));
         if nargin == 3 && isSave
            obj.Dy = exp(-x).*y.*y;
         end
      end   
   end
end

