classdef SoftmaxOutputLayer < StandardOutputLayer
   % TODO: improve robustness using "Tricks of the Trade" 2nd ed Ch 11
   % tricks

   methods
      function obj = SoftmaxOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, outputSize, varargin{:});
      end
      
      function [dLdz, y] = compute_dLdz(obj, x, t)
         y = obj.feed_forward(x);
         dLdz = y - t;
      end
      
      function y = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = softmax(z);
      end
   
      function loss = compute_loss(~, y, t)
         loss = mean(sum(-t.*log(y)));
      end 
   end
   
end

