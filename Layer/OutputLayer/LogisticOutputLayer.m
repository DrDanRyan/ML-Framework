classdef LogisticOutputLayer < StandardOutputLayer
   
   properties
      isLocalllyLinear = false
   end
   
   methods
      function obj = LogisticOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, outputSize, varargin{:});
      end
      
      function [dLdz, y] = compute_dLdz(obj, x, t)
         y = obj.feed_forward(x);
         dLdz = y - t;
         dLdz(isnan(t)) = 0;
      end
      
      function y = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = 1./(1 + exp(-z));
      end
      
      function value = compute_Dy(~, ~, y)
         value = y.*(1-y);
      end
      
      function value = compute_D2y(~, ~, y, Dy)
         value = Dy.*(1-2*y);
      end
   
      function loss = compute_loss(~, y, t)
         % Should be: loss = mean(-t.*log(y) - (1-t).*log(1-y))
         % This is more robust to NaN from 0*(-Inf) when t == y == 0 or t == y == 1
         loss = (sum(-log(y(t==1))) + sum(-log(1-y(t==0))))/length(t(~isnan(t)));               
      end
   end
   
end

