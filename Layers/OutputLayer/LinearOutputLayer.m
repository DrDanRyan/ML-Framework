classdef LinearOutputLayer < StandardOutputLayer
<<<<<<< HEAD
   % A linear layer with MeanSquaredError loss function. This ignores NaN values
   % in targets (useful for AutoEncoders when there are missing values).
   
=======
   % A linear layer with MeanSquaredError loss function. NaN values for targets
   % are ignored and not included in gradient or loss calculations.

>>>>>>> e540396056d196ecb0b1b604eb014cdcccd43daf
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
      
      function loss = compute_loss(~, y, t)
         loss = .5*nanmean(nansum((y-t).*(y-t), 1));
      end
   end
   
end

