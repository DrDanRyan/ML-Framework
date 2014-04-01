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
         [y, z] = obj.feed_forward(x);
         u = exp(-z)./(1 + exp(-z)); % u = 1 - y
         dLdz = obj.gpuState.zeros(size(y));
         idx = y<.5;
         dLdz(idx) = y(idx) - t(idx);
         dLdz(~idx) = 1 - t(~idx) - u(~idx);
         dLdz(isnan(t)) = 0;
      end
      
      function [y, z] = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = 1./(1 + exp(-z));
      end
      
      function Dy = compute_Dy(~, x, y)
         z = obj.compute_z(x);
         u = exp(-z)./(1 + exp(-z)); % u = 1-y
         Dy = y.*u;
      end
      
      function D2y = compute_D2y(~, ~, y, Dy)
         D2y = Dy.*(1-2*y);
      end
   
      function loss = compute_loss(~, y, t)       
         loss = mean(nansum(-t.*log(y) - (1 - t).*log(1 - y), 1), 2);
      end
   end
   
end
