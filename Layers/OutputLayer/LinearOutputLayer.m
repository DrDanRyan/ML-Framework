classdef LinearOutputLayer < StandardOutputLayer
   % A linear layer with MeanSquaredError loss function. This ignores NaN values
   % in targets (useful for AutoEncoders when there are missing values).

   methods
      function obj = LinearOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, outputSize, varargin{:});
      end
      
      function y = feed_forward(obj, x)
         y = obj.compute_z(x);
      end
      
      function [dLdz, y] = compute_dLdz(obj, x, t)
         y = obj.feed_forward(x);
         dLdz = 2*(y - t);
         dLdz(isnan(t)) = 0;
      end
      
      function loss = compute_loss(~, y, t)
         loss = nansum((y(:)-t(:)).^2)/size(y, 2);
      end
   end
   
end

