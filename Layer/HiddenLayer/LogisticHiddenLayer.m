classdef LogisticHiddenLayer < StandardHiddenLayer
   
   properties
      isLocallyLinear = false;
   end
   
   methods
      function obj = LogisticHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function y = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = 1./(1+exp(-z));
      end   
      
      function value = compute_Dy(~, ~, y)
         value = y.*(1-y);
      end
      
      function value = compute_D2y(~, ~, y)
         value = y.*(1-y).*(1-2*y);
      end
   end
end

