classdef SoftmaxOutputLayer < StandardOutputLayer
   
   methods
      function obj = SoftmaxOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, outputSize, varargin{:});
      end
      
      function [dLdz, y] = compute_dLdz(obj, x, t)
         y = obj.feed_forward(x);
         dLdz = y - t;
      end
      
      function value = compute_Dy(~, ~, ~)
         % Not implemented... need to return a matrix instead of a vector
         % because of cross-terms in the denominator sum of softmax?
         
         % should be yi*(1-yi) on diagonals and -yi*yj on off diagonals
      end
      
      function y = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = softmax(z);
      end
   
      function loss = compute_loss(~, y, t)
         loss = mean(sum(-t.*log(y), 1));
      end 
   end
   
end

