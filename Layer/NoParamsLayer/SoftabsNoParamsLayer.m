classdef SoftabsNoParamsLayer < NoParamsLayer
   
   properties
      epsilon
   end
   
   methods
      function obj = SoftabsNoParamsLayer(epsilon)
         if nargin < 1
            epsilon = 1e-8;
         end
         obj.epsilon = epsilon;
      end
      
      function y = feed_forward(obj, x, isSave)
         y = sqrt(x.*x + 1e-8);
         if nargin == 3 && isSave
            obj.Dy = x./y;
         end
      end
   end
   
end

