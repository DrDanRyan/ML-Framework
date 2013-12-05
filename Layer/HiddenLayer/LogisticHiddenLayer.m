classdef LogisticHiddenLayer < StandardHiddenLayer
   
   properties
      zVal
      isLocallyLinear = false
   end
   
   methods
      function obj = LogisticHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function y = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = 1./(1+exp(-z));
         
         if obj.isReuseVals
            obj.zVal = z;
         end
      end   
      
      function value = compute_Dy(~, x, y)
         if obj.isReuseVals
            z = obj.zVal;
         else
            z = obj.compute_z(x);
         end
         
         u = exp(-z)./(1 + exp(-z)); % u = 1-y
         value = y.*u;
      end
      
      function value = compute_D2y(~, ~, y, Dy)
         value = Dy.*(1-2*y);
      end
   end
end

