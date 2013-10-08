classdef SoftmaxOutputLayer < StandardOutputLayer
   
   properties
      nonlinearity = @softmax; % not actually used
   end
   
   methods
      function obj = SoftmaxOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, outputSize, varargin{:});
      end
      
      function [dLdz, y] = dLdz(obj, x, t)
         y = obj.feed_forward(x);
         dLdz = y - t;
      end
      
      function y = feed_forward(obj, x)
         y = softmax(bsxfun(@plus, obj.params{1}*x, obj.params{2}));
      end
   
      function loss = compute_loss(~, y, t)
         loss = mean(-t.*log(y));
      end 
   end
   
end

