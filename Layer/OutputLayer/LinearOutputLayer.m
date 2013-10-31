classdef LinearOutputLayer < StandardOutputLayer
   % A linear layer with MeanSquaredError loss function
   
   properties
      isLocallyLinear = true
      isDiagonalDy = true
   end
   
   methods
      function obj = LinearOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, outputSize, varargin{:});
      end
      
      function y = feed_forward(obj, x)
         y = obj.compute_z(x);
      end
      
      function [dLdz, y] = compute_dLdz(obj, x, t)
         y = obj.feed_forward(x);
         dLdz = y - t;
         dLdz(isnan(t)) = 0;
      end
      
      function value = compute_Dy(obj, ~, y)
         value = obj.gpuState.ones(size(y));
      end
      
      function value = compute_D2y(obj, ~, y, ~)
         value = obj.gpuState.zeros(size(y));
      end
      
      function loss = compute_loss(~, y, t)
         loss = .5*nanmean(nansum((y-t).*(y-t), 1));
      end
   end
   
end

