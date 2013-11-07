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
         one_minus_y = exp(-z)./(1 + exp(-z));
         dLdz = obj.gpuState.zeros(size(y));
         idx = y<.5;
         dLdz(idx) = y - t;
         dLdz(~idx) = 1 - t - one_minus_y;
         dLdz(isnan(t)) = 0;
      end
      
      function y = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = 1./(1 + exp(-z));
      end
      
      function value = compute_Dy(~, z, y)
         one_minus_y = exp(-z)./(1 + exp(-z));
         value = y.*one_minus_y;
      end
      
      function value = compute_D2y(~, ~, y, Dy)
         value = Dy.*(1-2*y);
      end
   
      function loss = compute_loss(~, y, t)       
         idx = y < .5;
         loss = t(:,idx).*log(y(:,idx)) + (1 - t(:,idx)).*log1p(-y(:,idx)) + ...  % could be improved by passing in z
                  t(:,~idx).*log1p(y(:,~idx)-1) + (1 - t(:,~idx)).*log(1-y(:,~idx));
      end
   end
   
end

