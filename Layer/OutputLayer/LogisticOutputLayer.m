classdef LogisticOutputLayer < StandardOutputLayer
   
   properties
      isLocallyLinear = false
      isDiagonalDy = true
   end
   
   methods
      function obj = LogisticOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, outputSize, varargin{:});
      end
      
      function [dLdz, y] = compute_dLdz(obj, x, t)
         z = obj.compute_z(x);
         y = 1./(1 + exp(-z));
         u = exp(-z)./(1 + exp(-z)); % u = 1 - y
         dLdz = obj.gpuState.zeros(size(y));
         idx = y<.5;
         dLdz(idx) = y - t;
         dLdz(~idx) = 1 - t - u;
         dLdz(isnan(t)) = 0;
      end
      
      function y = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = 1./(1 + exp(-z));
      end
      
      function value = compute_Dy(~, z, y)
         u = exp(-z)./(1 + exp(-z)); % u = 1-y
         value = y.*u;
      end
      
      function value = compute_D2y(~, ~, y, Dy)
         value = Dy.*(1-2*y);
      end
   
      function loss = compute_loss(~, y, t)       
         loss = -t.*log(y) - (1 - t).*log(1 - y);
      end
   end
   
end

