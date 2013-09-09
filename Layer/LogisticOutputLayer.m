classdef LogisticOutputLayer < StandardOutputLayer
   
   properties
      nonlinearity = @sigm;
   end
   
   methods
      function obj = LogisticOutputLayer(inputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, 1, varargin{:});
      end
      
      function value = dLdz(obj, x, t)
         y = obj.feed_forward(x);
         value = y - t;
      end
   
      function loss = compute_loss(~, y, t)
         loss = mean(-t.*log(y) - (1-t).*log(1-y));
      end
   end
   
end

