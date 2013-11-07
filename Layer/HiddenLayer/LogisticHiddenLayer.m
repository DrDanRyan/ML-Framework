classdef LogisticHiddenLayer < StandardHiddenLayer
   
   properties
      isLocallyLinear = false
   end
   
   methods
      function obj = LogisticHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function [y, z] = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = 1./(1+exp(-z));
      end   
      
      function value = compute_Dy(~, z, y)
         u = exp(-z)./(1 + exp(-z)); % u = 1-y
         value = y.*u;
      end
      
      function value = compute_D2y(~, ~, y, Dy)
         value = Dy.*(1-2*y);
      end
   end
end

