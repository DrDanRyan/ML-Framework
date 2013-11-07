classdef LogisticHiddenLayer < StandardHiddenLayer
   
   properties
      isLocallyLinear = false
      isDiagonalDy = true
      isRobust = true
   end
   
   methods
      function obj = LogisticHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function y = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = 1./(1+exp(-z));
      end   
      
      function value = compute_Dy(obj, x, y)
         if obj.isRobust
            z = obj.compute_z(x);
            one_minus_y = exp(-z)./(1 + exp(-z));
            value = y.*one_minus_y;
         else
            value = y.*(1-y);
         end
      end
      
      function value = compute_D2y(~, ~, y, Dy)
         value = Dy.*(1-2*y);
      end
   end
end

