classdef LogisticOutputLayer < StandardOutputLayer
   
   properties
      nonlinearity = @sigm;
   end
   
   methods
      function obj = LogisticOutputLayer(inputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, outputSize, varargin{:});
      end
      
      function [dLdz, y] = dLdz(obj, x, t)
         y = obj.feed_forward(x);
         dLdz = y - t;
         dLdz(isnan(t)) = 0;
      end
   
      function loss = compute_loss(~, y, t)
         % loss = mean(-t.*log(y) - (1-t).*log(1-y));
         loss = (sum(-log(y(t==1))) + sum(-log(1-y(t==0))))/length(t(~isnan(t))); % more robust to NaN from
                                                                       % 0*(-Inf) when t == y == 0
                                                                       % or t == y == 1              
      end
   end
   
end

