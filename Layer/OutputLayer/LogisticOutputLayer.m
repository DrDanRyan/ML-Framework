classdef LogisticOutputLayer < StandardOutputLayer
   
   properties
      isLocallyLinear = false
      isDiagonalDy = true
      isRobust = true
   end
   
   methods
      function obj = LogisticOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, outputSize, varargin{:});
      end
      
      function [dLdz, y] = compute_dLdz(obj, x, t)
         if obj.isRobust
            z = obj.compute_z(x);
            y = 1./(1 + exp(-z));
            one_minus_y = exp(-z)./(1 + exp(-z));
            dLdz = obj.gpuState.zeros(size(y));
            idx1 = y<.5;
            dLdz(idx1) = y - t;
            dLdz(~idx1) = 1 - t - one_minus_y;
         else
            y = obj.feed_forward(x);
            dLdz = y - t;
         end
         dLdz(isnan(t)) = 0;
      end
      
      function y = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = 1./(1 + exp(-z));
      end
      
      function value = compute_Dy(~, ~, y)
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
   
      function loss = compute_loss(~, y, t)
         % Should be: loss = mean(-t.*log(y) - (1-t).*log(1-y))
         % This is more robust to NaN from 0*(-Inf) when t == y == 0 or t == y == 1
         loss = (sum(-log(y(t==1))) + sum(-log(1-y(t==0))))/length(t(~isnan(t)));               
      end
   end
   
end

