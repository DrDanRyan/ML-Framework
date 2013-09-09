classdef SoftmaxOutputLayer < StandardOutputLayer
   
   properties
      nonlinearity = @softmax;
   end
   
   methods
      function obj = SoftmaxOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, outputSize, varargin{:});
      end
      
      function value = dLdz(obj, x, t)
         y = obj.feed_forward(x);
         value = y - t;
      end
   
      function loss = compute_loss(~, y, t)
         loss = mean(-t.*log(y));
      end 
   end
   
end

