classdef LinearOutputLayer < StandardOutputLayer
   % A linear layer with MeanSquaredError loss function
   
   properties
      nonlinearity = @(x) x; % not actually used, need to define to inherit from StandardOutputLayer
   end
   
   methods
      function obj = LinearOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, outputSize, varargin{:});
      end
      
      function value = feed_forward(obj, x)
         value = bsxfun(@plus, obj.params{1}*x, obj.params{2});
      end
      
      function [dLdz, y] = dLdz(obj, x, t)
         y = obj.feed_forward(x);
         dLdz = y - t;
      end
      
      function loss = compute_loss(~, y, t)
         loss = .5*mean(sum((y-t).*(y-t), 1));
      end
   end
   
end

